import os.path
import sys
import time
import string
from socket import *
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')


def text_process(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    test_list = text.split()
    test_list = [word.lower() for word in test_list]
    test_list = [word for word in test_list if word.lower().isalpha()]
    test_list = [word for word in test_list if word.lower() not in stopwords.words('english')]
    return " ".join(test_list)


def startClient(host, port, udp_port):
    client_socket = socket(AF_INET, SOCK_STREAM)
    client_socket.connect((host, port))
    print('Starting client on port:', udp_port)
    run = True
    while run:
        user_query = input("Please enter the content you want to search for (^exit for exit): ")
        if user_query == '':
            print("Please enter some words to start.")
            continue
        if user_query == '^exit':
            run = False
            continue
        query = text_process(user_query)
        query = query.encode('utf-8')
        client_socket.sendall(query)
        resp = client_socket.recv(1024 * 16).decode('utf-8')
        print(resp)
        filename = str(time.time()) + '.txt'
        with open(filename, 'w') as f:
            f.write("Query: {} \n".format(user_query))
            f.write(resp)
        print("Search result was saved in: ", filename)
    return


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python client.py server_address server_port udp_port")
    else:
        host = sys.argv[1]
        port = int(sys.argv[2])
        udp_port = int(sys.argv[3])
        startClient(host, port, udp_port)
        try:
            os._exit(0)
        except:
            print('Program is exited.')
