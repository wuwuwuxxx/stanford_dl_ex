function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
%% forward prop
%%% YOUR CODE HERE %%%
hAct{1} = sigmoid(bsxfun(@plus, stack{1}.W * data , stack{1}.b));
% hAct{2} = exp(stack{2}.W * hAct{1});
for j = 2:numHidden
    hAct{j} = sigmoid(bsxfun(@plus, stack{j}.W * hAct{j-1} , stack{j}.b));
end
hAct{numHidden+1} = exp(bsxfun(@plus, stack{numHidden+1}.W * hAct{numHidden}, stack{numHidden+1}.b));
hAct{numHidden+1} = bsxfun(@rdivide, hAct{numHidden+1}, sum(hAct{numHidden+1}));
%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  pred_prob = hAct{numHidden+1};
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
groundTruth = full(sparse(labels, 1:size(labels, 1), 1));
cost = -sum(sum(groundTruth .* log(hAct{numHidden+1})));
%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
deltaStack = cell(numHidden+1, 1);
deltaStack{numHidden+1} = -(groundTruth - hAct{numHidden+1});
for j = numHidden:-1:1
    deltaStack{j} = ((stack{j+1}.W)' * deltaStack{j+1}) .* (hAct{j}.*(1-hAct{j}));
end
% deltaStack{1} =  stack{2}.W' * deltaStack{2} .* (hAct{1} .* (1-hAct{1}));

% compute delta W and delta b
M = size(labels, 1);
gradStack{1}.W = deltaStack{1} * data';
gradStack{1}.b = sum(deltaStack{1}, 2);
for j = 2:numHidden+1
    gradStack{j}.W = deltaStack{j} * hAct{j-1}';
    gradStack{j}.b = sum(deltaStack{j}, 2);
end
% gradStack{2}.W = deltaStack{2} * hAct{1}' / M;
% gradStack{2}.b = mean(deltaStack{2}, 2);
%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
cost = cost + ei.lambda/2 * (norm(stack{1}.W, 'fro')^2 + norm(stack{2}.W, 'fro')^2);
for j = 1:numHidden+1
    gradStack{j}.W = gradStack{j}.W + ei.lambda * stack{j}.W;
end
% gradStack{1}.W = gradStack{1}.W + ei.lambda * stack{1}.W;
% gradStack{2}.W = gradStack{2}.W + ei.lambda * stack{2}.W;
%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



