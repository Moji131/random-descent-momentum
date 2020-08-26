from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


if rank != 0:
    data = rank
    comm.send(data, dest=0)
else:
    print("This is rank 0")
    f = open("out.txt", "w")
    f.write("This is rank 0")
    f.close()

    f = open("out.txt", "a")
    for i in range(1,size):
        data = comm.recv(source=i)
        str1 = "\nThis is a message from " + str(data)
        print(str1 )
        f.write(str1 )
    f.close()

