clear;
close all;

load data_test
load data_train
load labels_test
load labels_train

data_test = data_test';
data_train = data_train';
data_test = uint8(255*data_test);
data_train = uint8(255*data_train);
data_test = reshape(data_test, 10000, 1, 28, 28);
data_train = reshape(data_train, 60000, 1, 28, 28);

save lmdb

a = data_train(2,:,:,:);
a = reshape(a, 28, 28);
imshow(a);