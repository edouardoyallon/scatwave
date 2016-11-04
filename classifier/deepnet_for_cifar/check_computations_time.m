addpath_scatnet

N=[32 128 256];
J=[2 4 5];
for n=1:3
	for j=1:3
	[Wop,filters]=create_scattering_operators(N(n),N(n),'t',J(j));
	x=rand([N(n),N(n),3,128]);

	tic
           for p=1:10
               S=scat(x,Wop);
           end
	time=toc/10;
        fprintf('N: %i J: %f ; time : %f \n',N(n),J(j),time);
        end
end

    
