import socket
import sys

# In direct connection mode, the robot's default IP address is 192.168.2.1, and the control command port number is 40923
host = "192.168.2.1"
port = 40923

def main():

        address = (host, int(port))

        # Establish a TCP connection with the robot's control command port
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        print("Connecting...")

        s.connect(address)

        print("Connected!")

        while True:

                # Wait for the user to input a control command
                msg = input(">>> please input SDK cmd: ")

                # Exit the current program when the user enters Q or q
                if msg.upper() == 'Q':
                        break

                # Add a ';' terminator to the end
                msg += ';'

                # Transmit the control command to the robot
                s.send(msg.encode('utf-8'))

                try:
                        # Wait for the robot to return the execution result
                        buf = s.recv(1024)

                        print(buf.decode('utf-8'))
                except socket.error as e:
                        print("Error receiving :", e)
                        sys.exit(1)
                if not len(buf):
                        break

        # Disable the port connection
        s.shutdown(socket.SHUT_WR)
        s.close()

if __name__ == '__main__':
        main()