from hashlib import sha256
from mpmath import *

mp.prec = 300

result = []
data = bytearray()
with open('result.tsv') as f:
    assert f.readline().rstrip('\r\n') == 'rank\tuser_id\tpoint'
    for i, line in enumerate(f, start=1):
        rank, user_id, point = line.rstrip('\r\n').split('\t')
        assert int(rank) == i
        result.append(user_id)
        data += f'{user_id} {point}\n'.encode('utf-8')

key = sha256(data).hexdigest()

n = len(result)
p = '%064x' % int(ceil(atan(mpf(n-20) / mpf(10)) / mpf(10) * (1 << 256)))

print(key)
print(p)
num_win = 0
for user_id in result[20:]:
    digest = sha256(f'{user_id} {key}\n'.encode('utf-8')).hexdigest()
    win = num_win < 25 and digest < p
    if win:
        num_win += 1
    print(user_id, digest, win)
