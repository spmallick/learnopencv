import threading
import socket
import sys
import time
 
host = ''
port = 9000
locaddr = (host,port)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(locaddr)
tello_address = ('192.168.10.1', 8889)

def recv():
    count = 0
    while True:
        try:
            data, server = sock.recvfrom(1518)
            print(data.decode(encoding="utf-8"))
        except Exception:
            print ('\nExit . . .\n')
            Break
# Create a new thread for the recv function..recvThread = threading.Thread(target=recv)
recvThread.start()

while True:
    try:
        msg = input("")
        if not msg:
            break
        if 'end' in msg:
            print ('...')
            sock.close() 
            break
        # Send data.
        msg = msg.encode(encoding="utf-8")
        sent = sock.sendto(msg, tello_address)
    except KeyboardInterrupt:
        print ('\n . . .\n')
        sock.close() 
        break