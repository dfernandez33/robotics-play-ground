import torch
import tqdm
import numpy as np

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch import nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class EmbeddingModel(nn.Module):
    def __init__(self, GloVe, train, DIM_EMB=300, NUM_OUTPUTS=1, HID_DIM=100, FINAL_SOFTMAX=False):
        super(EmbeddingModel, self).__init__()
        self.DIM_EMB, self.HID_DIM, self.NUM_OUTPUTS, self.FINAL_SOFTMAX = DIM_EMB, HID_DIM, NUM_OUTPUTS, FINAL_SOFTMAX
        self.dataset_utils = train
        self.embedding, self.GloVe_inx = self.init_glove(GloVe)
        self.lstm = nn.LSTM(self.DIM_EMB, HID_DIM, bidirectional=True, batch_first=True)
        self.drop_layer = nn.Dropout()
        self.lin_layer = nn.Linear(2 * self.HID_DIM, self.NUM_OUTPUTS)
        self.log_probs = nn.LogSoftmax(dim=0)

    def init_glove(self, GloVe):
        tmp_dict = {self.dataset_utils.word2i[k]: v for k, v in GloVe.items()}
        GloVe_inx = sorted(tmp_dict.keys())
        GloVe_tensor = np.array([tmp_dict[v] for v in GloVe_inx])
        embeddings = torch.rand(len(self.dataset_utils.word2i) + 2, self.DIM_EMB).type(torch.FloatTensor)
        embeddings[GloVe_inx, :] = torch.from_numpy(GloVe_tensor).type(torch.FloatTensor)
        embeddings = nn.Embedding.from_pretrained(embeddings, freeze=False)
        return embeddings, GloVe_inx

    def forward(self, X):
        emb_words = self.embedding(X)
        rnn_hidden, (rnn_output, cell_state) = self.lstm(emb_words)
        score = self.lin_layer(rnn_hidden.sum(dim=1))
        if self.FINAL_SOFTMAX:
            return self.log_probs(score)
        else:
            return score

    def inference(self, X):
        X = X.split()
        X = self.dataset_utils._sentence2indx(X)
        X = X.to(device)
        return self.forward(X)


class BertTransformerVerbalReward(nn.Module):
    def __init__(self, model_path):
        super(BertTransformerVerbalReward, self).__init__()
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.bert_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1)
        if model_path:
            self.bert_model.load_state_dict(torch.load(model_path))

    def get_score(self, command):
        tkns = self.tokenizer(command)
        input = torch.tensor(tkns['input_ids']).unsqueeze(dim=0).to(device)
        attention_mask = torch.tensor(tkns['attention_mask']).unsqueeze(dim=0).to(device)
        return self.bert_model(input, attention_mask=attention_mask).logits.item()


def EvalNet(data_dev, train_utils, model):
    X, Y = train_utils.prepare_test(data_dev)
    X = X.to(device)
    Y = Y.to(device)
    Y_hat = model.forward(X)
    loss = nn.MSELoss(reduction='mean')
    return loss(Y_hat, Y)


def Train(model, data_loader, dev_data, train_utils, lr=0.001, num_epochs=15):
    print("Start Training!")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.1)
    criterion = nn.MSELoss(reduction='sum')
    clip = 50.0
    for epoch in tqdm.trange(num_epochs, desc="training", unit="epoch"):
        with tqdm.tqdm(
                data_loader,
                desc="epoch {}".format(epoch + 1),
                unit="batch",
                total=len(data_loader)) as batch_iterator:
            model.train()
            total_loss = 0.0
            for i, batch_data in enumerate(batch_iterator, start=1):
                X, Y = batch_data
                X = X.to(device)
                Y = Y.to(device).float()
                opt.zero_grad()
                Y_hat = model.forward(X)
                loss = criterion(Y_hat, Y)
                total_loss += loss.item()
                loss.backward()
                # Gradient clipping before taking the step
                _ = nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
                batch_iterator.set_postfix(mean_loss=total_loss / i, current_loss=loss.item(), lr=lr)
        sched.step()
        if epoch % 2 == 0:
            model.eval()
            eval_loss = EvalNet(dev_data[:], train_utils, model)
            print(f"Eval error: {eval_loss}")
