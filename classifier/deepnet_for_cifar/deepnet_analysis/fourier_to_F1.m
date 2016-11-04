function [inewW1]=fourier_to_F1(inewW1,p)
for c=1:3
    for i=1:4
        if(size(p{c}{i},1)==1)
        inewW1(:,:,p{c}{i},:)=ifft(inewW1(:,:,p{c}{i},:),[],3);
        
        else
            
            for t=1:8
                inewW1(:,:,p{c}{i}(:,t),:)=ifft(inewW1(:,:,p{c}{i}(:,t),:),[],3);
             end
            
            for t=1:8
                inewW1(:,:,p{c}{i}(t,:),:)=ifft(inewW1(:,:,p{c}{i}(t,:),:),[],3);
            end
            
             
                
        end
    end
end



inewW1=apply_DCT_along_first_inv(inewW1);
inewW1=permute(inewW1,[2 1 3 4]);
inewW1=apply_DCT_along_first_inv(inewW1);
inewW1=ipermute(inewW1,[2 1 3 4]);
end
