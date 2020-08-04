#!/usr/bin/env python

'''
get_agree_number.py
get agree number from the numbering server
'''

__author__  = 'Kohji'
__date__    = '2019-01-22'
__version__ = 0.1

__date__    = '2020-08-04'
__version__ = 0.2


def main():
  '''
  the main function
  '''

  import sys
  import random
  from socket import socket, AF_INET, SOCK_STREAM

  host     = 'localhost'
  port     = 56429	# port of the numbering server
  port_min = 57111	# min number of receiving port
  port_max = 57999	# max number of receiving port
  max      = 1024
  n_thread = 4

  socket_send = socket(AF_INET, SOCK_STREAM)	# sending
  socket_send.connect((host, port))
  socket_recv = socket(AF_INET, SOCK_STREAM)	# receiving
  while True:
    port = random.randint(port_min, port_max)	# creating receiving port number
    try:
      socket_recv.bind((host, port))
      socket_recv.listen(n_thread)
      break
    except:	# try again
      sys.stderr.write('Error 41: port ' + port + "\n")
  socket_send.send(str(port).encode('utf-8'))	# request to get a number using the number of receiving port
  socket_send.close()
  while True:
    try:
      (socket_prvd, address) = socket_recv.accept()	# numbering result
      message = socket_prvd.recv(max).decode('utf-8')	# get as a string
      socket_prvd.close()
      socket_recv.close()
      sys.stderr.write(message + "\n")
      break
    except:
      sys.stderr.write("Error 42\n")


if __name__ == '__main__':
  main()
