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
hAct{2} = exp(stack{2}.W * hAct{1});
% for j = 2:numHidden
%     hAct{j} = sigmoid(bsxfun(@plus, stack{j}.W * hAct{j-1} , stack{j}.b));
% end
% hAct{numHidden+1} = exp(stack{numHidden+1}.W * hAct{numHidden});
hAct{2} = bsxfun(@rdivide, hAct{2}, sum(hAct{2}));
%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  pred_prob = hAct{2};
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
groundTruth = full(sparse(labels, 1:size(labels, 1), 1));
cost = -mean(sum(groundTruth .* log(hAct{2})));
%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
deltaStack = cell(numHidden+1, 1);
deltaStack{2} = -(groundTruth - hAct{2});
% for j = numHidden:-1:1
%     deltaStack{j} = ((stack{j+1}.W)' * deltaStack{j+1}) .* (hAct{j}.*(1-hAct{j}));
% end
deltaStack{1} =  stack{2}.W' * deltaStack{2} .* (hAct{1} .* (1-hAct{1}));

% compute delta W and delta b
M = size(labels, 1);
gradStack{1}.W = deltaStack{1} * data' / M;
gradStack{1}.b = mean(deltaStack{1}, 2);
% for j = 2:numHidden+1
%     gradStack{j}.W = deltaStack{j} * hAct{j-1}' / size(labels, 1);
%     gradStack{j}.b = mean(deltaStack{j}, 2);
% end
gradStack{2}.W = deltaStack{2} * hAct{1}' / M;
gradStack{2}.b = mean(deltaStack{2}, 2);
%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



