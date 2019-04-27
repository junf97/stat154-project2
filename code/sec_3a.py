from sec_2a import split_data

if __name__ == '__main__':
    K = 5
    train_X, _, test_X, train_y, _, test_y = split_data(val_ratio=0, random_state=0)

