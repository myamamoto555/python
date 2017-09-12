
c = 0
with open("nips.txt") as f:
    for l in f:
        if c == 3277 or c == 2452:
            print l
        c += 1
