
if __name__ == "__main__":
    nums = [[1, 2, 3] * 2]
    with open("sc.txt", 'w+') as f:
        for row in nums:
            f.write(' '.join(map(str, row)) + '\n')