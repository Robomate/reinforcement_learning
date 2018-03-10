clc;
clear all; 
close all;
format short;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Value Iteration
% action order: (North, East, South, West)
% Roboball 8/2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% init globals
iterations = 100; %$ number of iterations
discount = 0.9; % discount factor
trans = [0.1,0.8,0.1]; % transitions probs for policy
arrow = {char(8593),char(8594),char(8595),char(8592)}; % actions
V_err_hist = zeros(iterations,1);
%arrow2 = ['up','right','down','left']; % actions
% init grid world
rows = 3;
cols = 4;
% Reward table
R = zeros(rows,cols);
R(1,4) = 1;
R(2,4) = -1;
disp('initial tables');
R
%R(10:20,40) = 2;
%R(10,24:34) = 1:0.2:3;
% Value Table
V = zeros(rows,cols)
% Policy Table
P = cell(rows,cols);

% start iterations
for iter = 1: iterations
disp('**************');
iter
V0 = V;    
% start recursion
for r=1:rows
    for c=1:cols
        % 4 corner cases first, then 4 rim cases, else standard case
        if r == 1 && c == 1; 
           % corner case, up-left
           V1 = [V0(r,c)  , V0(r,c)  , V0(r,c+1);
                 V0(r,c)  , V0(r,c+1), V0(r+1,c);
                 V0(r,c+1), V0(r+1,c), V0(r,c)  ;
                 V0(r+1,c), V0(r,c)  , V0(r,c)  ];
        elseif r == 1 && c == cols;
               % corner case, up-right
               V1  = [V0(r,c-1), V0(r,c)  , V0(r,c)  ;
                      V0(r,c)  , V0(r,c)  , V0(r+1,c);
                      V0(r,c)  , V0(r+1,c), V0(r,c-1);
                      V0(r+1,c), V0(r,c-1), V0(r,c)  ];          
        elseif r == rows && c == cols;
               % corner case, down-right
               V1  = [V0(r,c-1), V0(r-1,c), V0(r,c);
                      V0(r-1,c), V0(r,c)  , V0(r,c);
                      V0(r,c)  , V0(r,c)  , V0(r,c-1);
                      V0(r,c)  , V0(r,c-1), V0(r-1,c)];     
        elseif r == rows && c == 1;
               % corner case, down-left
               V1  = [V0(r,c)  , V0(r-1,c), V0(r,c+1); 
                      V0(r-1,c), V0(r,c+1), V0(r,c); 
                      V0(r,c+1), V0(r,c)  , V0(r,c); 
                      V0(r,c)  , V(r,c)  , V0(r-1,c)];       
        elseif r == 1 && c ~= 1 && c ~= cols;
               % rim case, up
               V1  = [V0(r,c-1), V0(r,c)  , V0(r,c+1); 
                      V0(r,c)  , V0(r,c+1), V0(r+1,c); 
                      V0(r,c+1), V0(r+1,c), V0(r,c-1); 
                      V0(r+1,c), V0(r,c-1), V0(r,c) ];          
        elseif c == cols && r ~= 1 && r ~= rows;
               % rim case, right       
               V1  = [V0(r,c-1) ,V0(r-1,c) , V0(r,c); 
                      V0(r-1,c) ,V0(r,c)   , V0(r+1,c); 
                      V0(r,c)   ,V0(r+1,c) , V0(r,c-1); 
                      V0(r+1,c) ,V0(r,c-1) , V0(r-1,c)];      
        elseif r == rows && c ~= 1 && c ~= cols;
               % rim case, down
               V1  = [V0(r,c-1), V0(r-1,c), V0(r,c+1); 
                      V0(r-1,c), V0(r,c+1), V0(r,c)  ; 
                      V0(r,c+1), V0(r,c)  , V0(r,c-1); 
                      V0(r,c)  , V0(r,c-1), V0(r-1,c)];         
        elseif c == 1 && r ~= 1 && r ~= rows;
               % rim case, left
               V1  = [V0(r,c)  , V0(r-1,c), V0(r,c+1); 
                      V0(r-1,c), V0(r,c+1), V0(r+1,c); 
                      V0(r,c+1), V0(r+1,c), V0(r,c)  ; 
                      V0(r+1,c), V0(r,c)  , V0(r-1,c)];    
        else
            % standard case, action order: (North, East, South, West)      
            V1  = [V0(r,c-1), V0(r-1,c), V0(r,c+1);
                   V0(r-1,c), V0(r,c+1), V0(r+1,c);
                   V0(r,c+1), V0(r+1,c), V0(r,c-1);
                   V0(r+1,c), V0(r,c-1), V0(r-1,c)];
        end
        r
        c
        VZ = V1
        % apply Bellman Update
        [M,I] = max(V1 * trans');  % get max value and best action
        V1;
        M
        I;
        V(r,c) = R(r,c) + discount * M; % Update Value Table
        P(r,c) = arrow(I); % Update Policy Table
    end
end
V_err = abs(sum(sum(V-V0))); % get convergence error
V_err_hist(iter,1) = V_err; % save error in history
end
% Print out results:
disp('final results');
V0 % check convergence
V % Value Table
P % Policy Table

% % value plot
% %figure(1);
% subplot(1,2,1)
% colormap jet
% h = cellplot(num2cell(uint8(V)));
% % h= surf(V);
% % ylabel('rows')
% % xlabel('cols')
% % zlabel('Value')
% title('Value Table')

% policy plot
%figure(2);
% subplot(1,2,2)
% cellplot(P)
% title('Policy Table')



% figure();
% plot(V_err_hist);
% title('Convergence Error');
% xlabel('iterations');
% ylabel('L1 error');
% 
% % plot      
% figure('Color', 'w', 'position', [10 10 600 400]);
% cellplot(P)  
% set(gca, 'XLim', [0 cols+1], 'YLim', [0 cols+1]);  
% axis square;  
% colormap summer
% title('Policy Table')
% 
% % plot      
% figure('Color', 'w');
% cellplot(num2cell(uint8(V))); 
% set(gca, 'XLim', [0 cols+1], 'YLim', [0 cols+1]);  
% axis square;  
% colormap summer  
% title('Value Table')



