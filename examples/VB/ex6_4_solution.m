import brml.*
import graphlayout.*
figure(1)
nstates=6;
[W X Y Z]=assign(nstates*ones(1,4)); % number of states of each variable
[w x y z]=assign(1:4);
a=25;
phi{1}=array([w x],rand([W X]).^a);
phi{2}=array([x y],rand([X Y]).^a);
phi{3}=array([y w],rand([Y Z]).^a);
phi{4}=array([z w],rand([Z W]).^a);
phi{5}=array([w y],rand([W Y]).^a);

p = condpot(multpots(phi)); % normalise so we can monitor the KL exactly

% approximating q structure:
qw=array(w,normp(rand([W 1])));
qx=array(x,normp(rand([X 1])));
qy=array(y,normp(rand([Y 1])));
qz=array(z,normp(rand([Z 1])));
xcord=[0.2 0.2 0.8 0.8]; ycord=[0.2 0.8 0.2 0.8];
subplot(1,2,1); draw_layout(markov(p)-eye(4),{'w','x','y','z'},zeros(4,1),xcord,ycord); title('original graph')
subplot(1,2,2); draw_layout(markov([qw qx qy qz])-eye(4),{'w','x','y','z'},zeros(4,1),xcord,ycord); title('Factorised graph')

error_a=zeros(1,length(5:5:25));
count=1;
for a=1:1:30

% set the potentials to random tables:
rand('state', 1606); %#ok<RAND>
randn('state', 1606); %#ok<RAND>
phi{1}=array([w x],rand([W X]).^a);
phi{2}=array([x y],rand([X Y]).^a);
phi{3}=array([y w],rand([Y Z]).^a);
phi{4}=array([z w],rand([Z W]).^a);
phi{5}=array([w y],rand([W Y]).^a);

p = condpot(multpots(phi)); % normalise so we can monitor the KL exactly

% approximating q structure:
qw=array(w,normp(rand([W 1])));
qx=array(x,normp(rand([X 1])));
qy=array(y,normp(rand([Y 1])));
qz=array(z,normp(rand([Z 1])));
%figure;
for loop=1:50
    ord=randperm(4);
    for o=ord
        switch o
            case 1
                qw = condpot(exppot(sumpot(multpots([logpot(p) qx qy qz]),w,0),1));
            case 2
                qx = condpot(exppot(sumpot(multpots([logpot(p) qw qy qz]),x,0),1));
            case 3
                qy = condpot(exppot(sumpot(multpots([logpot(p) qx qw qz]),y,0),1));
            case 4
                qz = condpot(exppot(sumpot(multpots([logpot(p) qx qy qw]),z,0),1));
        end
    end
    kl(loop) = KLdiv(multpots([qw qx qy qz]),p);

end
[margMF{1} margMF{2} margMF{3} margMF{4}]=assign([qw qx qy qz]);
%plot(kl,'-o'); title(sprintf('KL divergence when a=%d',a));drawnow


jpot=multpots(phi);
fprintf('\nExact and MF marginals:\n\n')
for i=1:length(potvariables(phi))
    fprintf('variable %d:\n    Exact     MF\n',i);
    disp([table(condpot(jpot,i)) table(margMF{i})])
    errMF(i)=mean(abs(table(condpot(jpot,i))-table(margMF{i})));

end
fprintf('when a=%d, mean error MF = %g\n',a,mean(errMF))
fprintf('when a=%d,  Kullback-Leibler Divergence = %d\n',a,min(kl));
error_a(1,count)=mean(errMF);
count=count+1;
%figure; t=table(multpots(p)); plot(t(:),'-o'); title(sprintf('p distribution when a=%d',a))
end
plot(error_a)
