
function [newW1,p]=W1_to_fourier(filters_1)
for c=1:3
    
    start=(c-1)*size(filters_1,3)/3;
    % x*phi
    p{c}{1}=start+1;
    % |x*psi_1|*phi
        for theta1=1:8
            p{c}{2}(theta1)=start+1+1+(theta1-1)*9;
        end
         % |x*psi_2|*phi
         for theta1=1:8
            p{c}{3}(theta1)=start+1+8*9+theta1;
        end
% ||x*psi_1|*psi_2|*phi        
         for theta1=1:8
          for theta2=1:8
              
              
              theta1c=1+mod(theta1-1,8);
              theta2c=1+mod(theta2-1,8);
              %f(mod(theta2-theta1+1,8)-1,mod(theta1-1,8)+1 )=theta2
             p{c}{4}(mod(theta2-theta1-1,8)+1,mod(theta1-1,8)+1)=start+1+1+(theta1c-1)*9+theta2c;
            % p{c}{4}(mod(theta1-1,8)+1,mod(theta2-1,8)+1)=start+1+1+(theta1c-1)*9+theta2c;
            
          end
         end
        
        
end

%     l=[];
% for c=1:3
%     for i=1:4
%     l=[l;p{c}{i}(:)];
%     end
% end

%for c=1:3
%    for l=1:4
newW1=filters_1;

% Put it in Haar
newW1=apply_haar_along_first(newW1);
newW1=permute(newW1,[2 1 3 4]);
newW1=apply_haar_along_first(newW1);
newW1=ipermute(newW1,[2 1 3 4]);
for c=1:3
    for i=1:4
        if(size(p{c}{i},1)==1)
        newW1(:,:,p{c}{i},:)=fft(newW1(:,:,p{c}{i},:),[],3);
        
        else
            
            for t=1:8
                newW1(:,:,p{c}{i}(t,:),:)=fft(newW1(:,:,p{c}{i}(t,:),:),[],3);
            end
            
             for t=1:8
                newW1(:,:,p{c}{i}(:,t),:)=fft(newW1(:,:,p{c}{i}(:,t),:),[],3);
             end
                
        end
    end
end
end

