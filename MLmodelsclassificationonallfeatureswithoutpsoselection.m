%-------------------------------------------------------------------%
% Particle Swarm Optimization (PSO) source codes demo version      %
%-------------------------------------------------------------------%

%---Inputs-----------------------------------------------------------
% feat     : feature vector ( Instances x Features )
% label    : label vector ( Instances x 1 )

%---Outputs-----------------------------------------------------------
% C_knn_all: Confusion Matrix for KNN (All Features)
% C_svm_all: Confusion Matrix for SVM (All Features)
% C_tree_all: Confusion Matrix for Decision Tree (All Features)
% C_nb_all: Confusion Matrix for Naive Bayes (All Features)
%---------------------------------------------------------------------

%% Load data
load deeplemonTestingfeaturesandlabels.mat;

% Set 20% data as validation set
ho = 0.2;

% Hold-out method
HO = cvpartition(labels,'HoldOut',ho);

% Obtain training and testing data
X_train = features(HO.training,:);
Y_train = labels(HO.training,:);
X_test = features(HO.test,:);
Y_test = labels(HO.test,:);

% Train KNN model on all features
model_knn_all = fitcknn(X_train, Y_train);
predictions_knn_all = predict(model_knn_all, X_test);
C_knn_all = confusionmat(Y_test, predictions_knn_all);

% Train SVM model on all features
model_svm_all = fitcsvm(X_train, Y_train);
predictions_svm_all = predict(model_svm_all, X_test);
C_svm_all = confusionmat(Y_test, predictions_svm_all);

% Train Decision Tree model on all features
model_tree_all = fitctree(X_train, Y_train);
predictions_tree_all = predict(model_tree_all, X_test);
C_tree_all = confusionmat(Y_test, predictions_tree_all);

% Train Naive Bayes model on all features
model_nb_all = fitcnb(X_train, Y_train);
predictions_nb_all = predict(model_nb_all, X_test);
C_nb_all = confusionmat(Y_test, predictions_nb_all);

% Display confusion matrices for all features without PSO selection
disp('Confusion Matrix for KNN (All Features):');
disp(C_knn_all);

disp('Confusion Matrix for SVM (All Features):');
disp(C_svm_all);

disp('Confusion Matrix for Decision Tree (All Features):');
disp(C_tree_all);

disp('Confusion Matrix for Naive Bayes (All Features):');
disp(C_nb_all);
