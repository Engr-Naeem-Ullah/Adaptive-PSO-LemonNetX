clc;
clear all;

% Assuming you have extracted features (X) from CNN and corresponding labels (Y)
% X: N x D matrix (N samples, D features)
% Y: N x 1 vector of labels
% Load labels and features
load('label.mat'); % Load labels
load('sFeat.mat'); % Load features

load('deeplemonnet.mat')
net=deeplemonnet;
% Load testing data
load('deeplemonTfeatures.mat');
X_test = deeplemonTfeatures;

% Combine labels and features
X = sFeat;
Y = label;

% Train Decision Tree
tree = fitctree(X, Y);

% Visualize Decision Tree (Optional)
view(tree, 'Mode', 'graph');

% Extract rules from the decision tree
rules = extractRules(tree, net); % Custom function to extract rules

% Display extracted rules
disp('Extracted Rules:');
disp(rules);

% Save the rules to a file
save('extracted_rules.mat', 'rules');


function rules = extractRules(tree, net)
    % Initialize cell array to store rules
    rules = {};

    % Recursively traverse the decision tree
    rule = '';
    extractRulesRecursive(tree, 1, rule, net, rules);
end

function extractRulesRecursive(tree, node, rule, net, rules)
    % If leaf node is reached
    if tree.Children(node, 1) == 0 && tree.Children(node, 2) == 0
        % Extract rule
        class = tree.ClassNames{mode(tree.Y)};
        rule = [rule ' => ' class];
        rules{end+1} = rule;
    else
        % If not a leaf node
        if ~isempty(rule)
            rule = [rule ' & '];
        end
        feat = tree.CutPredictor{node};
        val = tree.CutPoint(node);
        rule = [rule feat ' <= ' num2str(val)];
        
        % Recursively traverse left child
        if tree.Children(node, 1) ~= 0
            extractRulesRecursive(tree, tree.Children(node, 1), rule, net, rules);
        end
        
        % Recursively traverse right child
        if tree.Children(node, 2) ~= 0
            % Negate the condition for the right child
            rule = strrep(rule, ' & ', ' & ~');
            extractRulesRecursive(tree, tree.Children(node, 2), rule, net, rules);
        end
    end
end