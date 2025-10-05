
% Read data from a text file
gt = readtable('CompetitionTestMap2.txt'); % Assuming the file has columns 'x' and 'y'
pp_path = readtable('CompetitionTestMap2_pp');
% Extract x and y coordinates
x1 = gt.Var1; % Replace 'x' with the actual column name if different
y1 = gt.Var2; % Replace 'y' with the actual column name if different

x2 = pp_path.Var1;
y2 = pp_path.Var2;

% Create a figure
figure;
plot(x1, y1, 'b.','DisplayName','gt'); % Plot with circles at data points and lines connecting them
hold on;
plot(x2,y2,'r.','DisplayName','PPpath');
hold off;

legend show;
xlabel('X Coordinate');
ylabel('Y Coordinate');
title('Plot of X and Y Coordinates');
grid on;