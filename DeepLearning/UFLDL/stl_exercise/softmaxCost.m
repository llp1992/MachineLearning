function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
% 10x8
theta = reshape(theta, numClasses, inputSize);
%100  data 8x100
numCases = size(data, 2);
%sparse(r,c,v) =  [r(i),c(i)] = v(i)
% 每一列的第 label 行 设置为i，也就是，每个样本属于哪个label，该位置就是1，其他的均为0
% 10 x 100
groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;
%10 x 8
thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

% 10x100  10x8  8x100   100个样本，每个样本属于每个label的概率
M = theta * data;
M = exp(bsxfun(@minus,M,max(M,[],1)));
H = bsxfun(@rdivide,M,sum(M));
cost = - sum(sum((groundTruth .* log(H)))) / size(data,2) + lambda * sum(sum(theta.^2)) / 2 ;
% 10x8      10x100   100x8
thetagrad = - ((groundTruth - H)) * data' /size(data,2) +lambda * theta;









% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

