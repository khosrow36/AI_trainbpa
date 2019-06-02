clear; 
close all; 
clc;
format compact;

DValue = importdata('glass.data', ',');
Value2=DValue(:,2:10);
y2 = DValue(:,11);


Pn = Value2;
T = y2;

Pn = transpose(Pn);
T = T';

Pn = mapminmax(Pn);
T=mapminmax(T);

trainFcn = 'traingda';  

S1_vec = 1:1:20; %neurony dla 1 warstwy
S2_vec = S1_vec; %neurony dla 2 warstwy
lr_inc_vec = 1:0.01:1.09;
lr_dec_vec = 0.5:0.1:0.9;
e_vec = 1:0.01:2;

verse = 1; %wiersz w pliku z wynikami
todo = length(S1_vec)*length(S2_vec); %obliczenie, zeby w pliku z parametrami tyle wierszy ile eksperymentow
parameters_pk = zeros(todo,3); %wyzerowanie pliku parameters_pk
parameters_sse = parameters_pk;
parameters_e = parameters_pk;

PK_v=zeros (length(S1_vec),length(S2_vec));
SSE_v=PK_v;
epoch_best_v = PK_v;
epoch_num_v = PK_v;

for e_ind = e_vec
%for lr_inc = lr_inc_vec
         %for lr_dec = lr_dec_vec  

%for S1 = S1_vec 
         %for S2 = S2_vec:S1  
                  %for ind_lr_inc=1:length(lr_inc_vec)
                          % for ind_lr_dec=1:length(lr_dec_vec)
                                    % for ind_e=1:length(e_vec)
                                        % Create a Pattern Recognition Network
                                        %hiddenLayerSize = [S1,S2];
                                        hiddenLayerSize = [2,7];
                                        net = feedforwardnet(hiddenLayerSize, trainFcn);
                                        net.performFcn = 'sse';
                                        net.layers{1}.transferFcn='tansig';
                                        net.layers{2}.transferFcn='tansig';
                                        net.layers{3}.transferFcn='purelin';

                                         %net = fitnet([hiddenLayerSize, hiddenLayer2Size],trainFcn);
                                        net.trainParam.epochs = 10000;
                                        net.trainParam.max_fail = 50;
                                        net.trainParam.lr = 0.01; 
                                        net.divideFcn = 'dividetrain';
                                        net.trainParam.goal = 0.25;
                                        net.trainParam.max_perf_inc = e_ind;
                                        %net.trainParam.lr_inc = lr_inc; 
                                        %net.trainParam.lr_dec = lr_dec;
                                      %  net.trainParam.max_perf_inc = e_vec(ind_e);

                                        % Setup Division of Data for Training, Validation, Testing
                                        %net.divideParam.trainRatio = 90/100;
                                        %net.divideParam.valRatio = 5/100;
                                        %net.divideParam.testRatio = 5/100;

                                        % Train the Network
                                        [net,tr] = train(net,Pn,T);
                                        verse = verse +1;

                                        % Test the Network
                                        y = net(Pn);
                                        e = gsubtract(T,y);
                                        performance = perform(net,T,y);
                            
    
                                        PK = (1-sum(abs(T-y)>=.5)/length(T))*100 %ogolny wzor na poprawnosc klasyfikacji
                       
                                        %parameters_pk(verse, 1) = lr_inc;
                                        %parameters_pk(verse, 2) = lr_dec;
                                        %parameters_pk(verse, 3) = PK;
                                        
                                        parameters_e(verse, 1) = e_ind;
                                        parameters_e(verse, 2) = PK;
                                        parameters_e(verse, 3) = tr.best_perf;
                                        
                                        %parameters_sse(verse, 1) = lr_inc;
                                        %parameters_sse(verse, 2) = lr_dec;
                                        %parameters_sse(verse, 3) = tr.best_perf;
                                        
                                        %PK_v(S1, S2) = PK;
                                        %SSE_v(S1, S2) = tr.best_perf;
                                        %epoch_best_v(S1,S1) = tr.best_epoch;
                                        %epoch_num_v(S1,S2) = tr.num_epochs;

                                        
                                        %PK_v(S1, S2, ind_lr_inc, ind_lr_dec, ind_e) = PK;
                                        %SSE_v(S1, S2, ind_lr_inc, ind_lr_dec, ind_e) = sse_value;
                                       % LR_v(S1, S2, ind_lr_inc, ind_lr_dec, ind_e) = net.trainParam.lr;
                                        %EPOCH_v(S1, S2, ind_lr_inc, ind_lr_dec, ind_e) = tr.best_epoch;
        
                                     %end
                          % end
                 % end
         %end
end
% View the Network
view(net)

% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
%figure, plotconfusion(T,y)
%figure, plotroc(T,y)
save("output14.mat", 'parameters_e')
