import random
import numpy as np
import math
from bitstring import BitArray


def codeByHamming(generated_sign_whole,Nsim,n,k):
    
    #kodiranje signala preko Hammingovog koda

    generated_sign_part = [0] * k
    m = 0;
    i = 0;
    coded_sign_whole = [];

    #cepamo signal na delove od po k

    while i<Nsim:  
        while m<k:
            generated_sign_part[m] = generated_sign_whole[i];
            m = m+1;
            i = i+1;
        

        coded_sign_part = hammingEncode(n,k,generated_sign_part);
        m=0;
        coded_sign_whole.extend(coded_sign_part)

    return coded_sign_whole

def decodeByHamming(coded_sign_whole,Nsim,n,k):
    
    #dekodiranje signala preko Hammingovog koda

    coded_sign_part = [0] * n
    m = 0;
    i = 0;
    recieved_sign_whole = [];
    new_size = (Nsim/k)*n 
    while i<new_size:  
        while m<n:
            coded_sign_part[m] = coded_sign_whole[i];
            m = m+1;
            i = i+1;
        

        recieved_sign_part = hammingDecode(n,k,coded_sign_part);
        m=0;
        recieved_sign_whole.extend(recieved_sign_part)

    return recieved_sign_whole

    
def repetitionEncode(generated_sign_whole, order):

    coded_sign_whole = [0] * (len(generated_sign_whole)*order)
    
    i = 0
    m = 0
    while i<(len(coded_sign_whole)):
        coded_sign_whole[i] = generated_sign_whole[m]
        coded_sign_whole[i+1] = generated_sign_whole[m]
        coded_sign_whole[i+2] = generated_sign_whole[m]
        i = i+3
        m = m+1
    return coded_sign_whole

def repetitionDecode(coded_sign_whole,order):
    i = 0;
    m = 0;
    recieved_sign_whole = [0]* int(len(coded_sign_whole)/3);
    while i < len(coded_sign_whole):
        sum =0;
        for j in range(order):  #majoriity vote
            sum = sum + coded_sign_whole[i+j];
        
        if sum >=2:
            recieved_sign_whole[m] = 1;
        else:
            recieved_sign_whole[m] = 0;
        m = m+1;
        i = i +3;
        
    return recieved_sign_whole

 
def submatrix (n,k):

  #pravi matricu binarnih predstava onih brojeva koji nisu stepen dvojke
 
        parity_num = n-k;
        A = []
        m= 0
        #izdvajamo one brojeve cije se binarne predstave ne nalaze u jedinicnoj matrici
        for i in range(1,n+1):
            p = math.pow(2,m)
            if i==p:
                m= m+1
            else:
                A.append(i) 
        A.sort(reverse=True)
        B = np.zeros((k,parity_num));

       
        for i in range( 0, k):
            for j in range( 0,parity_num):
                pera =int(math.pow(2,j)) & int(A[i])
                if pera :
                    B[i,parity_num-j-1] = 1;
                else:
                    B[i,parity_num-j-1] = 0
        
        return B
      

def errBurst(coded_sign_whole,burst_len):
    #greska koja se javlja u paketima duzine burst_len sa ucestanoscu burst_freq

    burst_freq = 40/burst_len
    transfer_sign_whole  = coded_sign_whole;
    sign_len = len(coded_sign_whole);
    random_err_position = []

    #delimo signal na onoliko dellova kolika je ucestanost greske 
    # u jednom delu izaberemo random poziciju greske 

    for i in range(0,sign_len,int(sign_len/burst_freq)): 
        random_err_position.append(random.randint(i, i+sign_len/burst_freq-(burst_len-1)))


    
    for i in range(len(random_err_position)):
        for j in range (0,burst_len):
            transfer_sign_whole[random_err_position[i]+j] = not coded_sign_whole[random_err_position[i]+j]; 
     
    return transfer_sign_whole



def errProbability (coded_sign_whole,err_prob):

    #kreiranje greske odredjene verovatnocom iste
    #kreiranje signala sa greskom
   
    transfer_sign_whole  = np.zeros(len(coded_sign_whole));

    num_err_bits = 0
    index_err_bits = [] 
    
    for i in range(len(coded_sign_whole)):

       m= random.random()   
       if m <=err_prob:
           num_err_bits = num_err_bits+1
           transfer_sign_whole[i] = not coded_sign_whole[i]
           
           index_err_bits.append(i)
       else:
           transfer_sign_whole[i] = coded_sign_whole[i];        
    
    # da znamo ako je netacno da nije nasa greska
    print("broj bitova sa greskom:", num_err_bits)
    print("indeks bitova u kodiranom signalu na kojima je greska:", index_err_bits)
    return transfer_sign_whole

def generator(Nsim, P0):

   
    generated_sign_whole = np.zeros(Nsim)
    for i in range(Nsim):
        m= random.random()   
        if m <=P0:

            generated_sign_whole [i] = 0;
        else:
            generated_sign_whole[i] = 1;
        
     

    return generated_sign_whole

def hammingEncode(n,k,generated_sign_part):

    B = submatrix (n,k);  
    
    #Generatorska matrica
    Identity = np.identity(k) 
    G = np.concatenate((Identity,B), axis=1)
    #ovaj deo je potreban da bi indeksi bili redom rasporedjeni
    if(n==12):
         p8 = G[:,8]
         G = np.delete(G, 8, 1)
         G = np.insert(G, 4, p8, axis=1)
         G[:,[8, 9]] = G[:,[9, 8]]
    if n==7:
        G[:,[3, 4]] = G[:,[4, 3]]


    # kodiranje poruke
    result = np.matmul(generated_sign_part, G)
    coded_sign_part = result%2; 
    return coded_sign_part

def hammingDecode(n,k,coded_sign_part):
    
    B = submatrix (n,k);
    Identity = np.identity(n-k)
    H = np.concatenate((B, np.identity(n-k)))     #Parity-check matrica

    # d7 d6 d5 p4 d3 p2 p1
    #zbog oblika matrice kada je koji 
    if(n==12):
         p8 = H[8]
         H = np.delete(H, 8, 0)
         H = np.insert(H, 4, p8, axis=0)
         H[[8, 9]] = H[[9, 8]]
    if n==7:
        H[[3, 4]] = H[[4, 3]]



        
    mika = np.matmul( coded_sign_part,H)

    syndrome = mika%2;                    
    
    position = BitArray(syndrome)
    pera= position.uint
    
  
    #pera govori indeks greske

    if pera:
     
        coded_sign_part[n-(pera)] =  1 -  coded_sign_part[n-(pera)];
   
   
    coded_sign_part = np.flip(coded_sign_part, 0)
   
    recived_sign_part = []
    m= 0
    #izdvajamo one brojeve cije se binarne predstave ne nalaze u jedinicnoj matrici
    #jer su oni deo originalnog signala
    for i in range(1,n+1):
        p = math.pow(2,m)
        if i==p:
            m= m+1
        else:
            recived_sign_part.append(coded_sign_part[i-1]) 
        
    recived_sign_part = np.flip(recived_sign_part, 0)
        
    return recived_sign_part

def interleave(input_sequence, word_length, no_of_words):

    output_sequence = [0] * len(input_sequence);
    
    if len(input_sequence)% word_length != 0:
        return -1
   
    
    remaining_words = int(float(len(input_sequence)) / word_length)% no_of_words; # dali ce biti neka rec van matrice
    no_of_passes = int(float(int(float(len(input_sequence))) / word_length) / no_of_words); #broj matrica
    
    for rep in range (0,no_of_passes):
        iN = [0] * ( word_length*no_of_words) 
        ouT = [0] * ( word_length*no_of_words)
    
        #interleaving matrix
        matrix =np.zeros((no_of_words, word_length))
        
        #construct word block to interleave
        for pos in range( 0,len(iN)):
            iN[pos] = input_sequence[rep*word_length*no_of_words + pos];
       
        
        #%construct matrix in row-first order
        for ct1 in range (0,no_of_words):
            for ct2 in range( 0,word_length):
                matrix[ct1, ct2] = iN[ct1*word_length + ct2];
         
   
        
        #%read matrix in column first order
        for ct1 in range( 0,word_length):
            for ct2 in range (0,no_of_words):
                ouT[ct1*no_of_words + ct2] = matrix[ct2, ct1];
           
       
        
        #copy word block to output sequence
        for pos in range( 0,len(ouT)):
            output_sequence[rep*word_length*no_of_words + pos] = ouT[pos];
       
   
    
    #interleave any remaining words with a reduced size matrixž
    #ovaj deo izaziva ostatak koji pri deinterlivigu izlazi iz funkcije izbegnuti ga
    if remaining_words:
        matrix = np.zeros((remaining_words, word_length));
    
        #%construct matrix in row-first order
        for ct1 in range (0,remaining_words):
            for ct2 in range( 0,word_length):
                zika =len(input_sequence) - (remaining_words*word_length)
                matrix[ct1, ct2] = input_sequence[len(input_sequence) - (remaining_words*word_length) + ct1*word_length  + ct2]
        
   
        #read matrix in column first order and copy to output
        for ct1 in range (0,word_length):
            for ct2 in range( 0,remaining_words):
                id = len(input_sequence) - (remaining_words*word_length)  + ct1*remaining_words + ct2
                output_sequence[id] = matrix[ct2, ct1]
       
   

    return output_sequence

def main ():

    Nsim = 20;
    P0 = 0.5;


    err_prob = 0.1;
    
    burst_len = 4;
    burst_freq = 10;
    order = 3;

    word_length = 7 
    no_of_words = 5

    #generated_sign_whole = generator(Nsim,P0) ## generisan signal na pocetku
    #coded_sign_whole = codeByHamming(generated_sign_whole,Nsim,12,8);  ## vracen ceo kodiran signal
    #transfered_sign_whole = errBurst (coded_sign_whole,burst_len) ##pri prenosenju dolazi do greske
    #transfered_sign_whole = errProbability (coded_sign_whole,err_prob)
    #recieved_sign_whole = decodeByHamming(transfered_sign_whole,Nsim,12,8);  ## vracen ceo kodiran signal

    H7generated_sign_whole = generator(Nsim,P0) ## generisan signal na pocetku
    H7coded_sign_whole = codeByHamming(H7generated_sign_whole,Nsim,7,4);  ## vracen ceo kodiran signal
    H7interleaved= interleave(H7coded_sign_whole,word_length,no_of_words)
    H7transfered_sign_whole = errProbability (H7interleaved,err_prob) ##pri prenosenju dolazi do greske
    H7deinterleaved= interleave(H7transfered_sign_whole,no_of_words,word_length)
    H7recieved_sign_whole = decodeByHamming(H7deinterleaved,Nsim,7,4);  ## vracen ceo kodiran signal

    #Rgenerated_sign_whole = generator(Nsim,P0) ## generisan signal na pocetku
    #Rcoded_sign_whole = repetitionEncode(Rgenerated_sign_whole,3);  ## vracen ceo kodiran signal
    #Rtransfered_sign_whole = errProbability (Rcoded_sign_whole,err_prob) ##pri prenosenju dolazi do greske
    #Rrecieved_sign_whole = repetitionDecode(Rtransfered_sign_whole,3);  ## vracen ceo kodiran signal

    
    generated_sign_whole_int = [int(i) for i in H7generated_sign_whole]  
    print("poslat signal:", generated_sign_whole_int)
    recieved_sign_whole_int = [int(i) for i in H7recieved_sign_whole]
    print("primljen signal",recieved_sign_whole_int )


    
if __name__ == "__main__":
    main()






