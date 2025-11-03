% Suppose you already have true labels and scores:
% [X,Y,T,AUC] = perfcurve(labels,scores,posclass);

figure;
plot(X, Y, 'b-', 'LineWidth', 2); % Thick blue ROC line
hold on;
plot([0 1], [0 1], 'k--', 'LineWidth', 1.5); % Dashed black diagonal
hold off;

xlabel('False Positive Rate', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('True Positive Rate', 'FontSize', 12, 'FontWeight', 'bold');
title('ROC Lemon Classification', 'FontSize', 14, 'FontWeight', 'bold');

grid on;
set(gca, 'FontSize', 12, 'LineWidth', 1.2);

% Save high-quality figure
print('ROC_Lemon_HQ','-dpng','-r600');  % 600 DPI high-res PNG
