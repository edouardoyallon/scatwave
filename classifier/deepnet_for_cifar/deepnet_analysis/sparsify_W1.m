filters_1=load('../operator_W1.mat');%,'x');
W1=filters_1.x;

[W1_f,p]=W1_to_fourier(W1);
sum(abs(W1_f(:))>0)/numel(W1_f)
%W1_f=(abs(W1_f)>0.11).*W1_f;

 for i=1:3
    W1_f(:,:,p{i}{2}(3:7),:)=0; 
    W1_f(:,:,p{i}{3}(3:7),:)=0;
    W1_f(:,:,p{i}{4}(3:7,:),:)=0;
    W1_f(:,:,p{i}{4}(:,3:7),:)=0;
 end
% 
% W1_f(2:end,:,:,:)=0;
% W1_f(:,2:end,:,:)=0;


sum(abs(W1_f(:))>0)/numel(W1_f)
W1_p=fourier_to_W1(W1_f,p);
save('../operator_sparsified_W1.mat','-v7.3','W1_p')