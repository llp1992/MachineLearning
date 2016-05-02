function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)
%lambda = 0;
%beta = 0;
% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 

% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

% 学习率 自己定义的
alpha = 0.03;

% 计算隐藏层神经元的激活度
p = zeros(hiddenSize,1);

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 
numPatches = size(data,2);
KLdist = 0;

%% 向前传输

a2 = sigmoid(W1*data+repmat(b1,[1,numPatches]));
p = sum(a2,2);
a3 = sigmoid(W2 * a2 + repmat(b2,[1,numPatches]));
J_sparse = 0.5 * sum(sum((a3-data).^2));

%% 计算 隐藏层的平均激活度
p = p /  numPatches ;

%% 向后传输 

    residual3 = -(data-a3).*a3.*(1-a3);
    tmp = beta * ( - sparsityParam ./ p + (1-sparsityParam) ./ (1-p));
    residual2 = (W2' * residual3 + repmat(tmp,[1,numPatches])) .* a2.*(1-a2);
    
    W2grad = residual3 * a2' / numPatches + lambda * W2 ;
    W1grad = residual2 * data'  / numPatches + lambda * W1 ;
    b2grad = sum(residual3,2) / numPatches; 
    b1grad = sum(residual2,2) / numPatches; 

%% 更新权重参数   加上 lambda  权重衰减
W2 = W2 - alpha * ( W2grad  );
W1 = W1 - alpha * ( W1grad );

b2 = b2 - alpha * (b2grad );
b1 = b1 - alpha * (b1grad );

%% 计算KL相对熵
for j = 1:hiddenSize
    KLdist = KLdist + sparsityParam *log( sparsityParam / p(j) )   +   (1 - sparsityParam) * log((1-sparsityParam) / (1 - p(j)));
end

%% costFunction 加上 lambda 权重衰减
cost = J_sparse / numPatches + (sum(sum(W1.^2)) + sum(sum(W2.^2))) * lambda / 2  + beta * KLdist;

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end


