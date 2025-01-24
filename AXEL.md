Each record in the list has:
12-byte x-coordinate (compressed secp256k1 point)
19-byte distance value
1-byte type (all are type 0 = tame in this file)

To get the private key from a collision between a tame and wild point:
For tame points (type=0):
privKey = tame_distance - wild_distance + Int_HalfRange
For wild points (type=1,2):
privKey = (tame_distance - wild_distance)/2 + Int_HalfRange

To verify a solution:
Take the private key
Multiply the secp256k1 generator point G by the private key
Check if the resulting point's x-coordinate matches the record
The records are organized in a hash table structure where:

The first 3 bytes of the x-coordinate are used as the hash key [i][j][k]
All points that share those first 3 bytes are stored in the same list
The largest list in this file has 43 points, all sharing the prefix [00 f1 f5]
