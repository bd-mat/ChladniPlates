n = 100;      % # of points per direction
mu = 0.3;   % poisson's ratio
h = 2/(n-3);% intergrid distance
M = 50;     % # of desired eigenvalues

% initialise grid
G = numgrid('S',n+2); D = delsq(G);
% define bounds + ghosts
bl = G(3:n,3); br = G(3:n,n); bt = G(3,3:n)'; bb = G(n,3:n)';
gl = G(3:n,2); gr = G(3:n,n+1); gt = G(2,3:n)'; gb = G(n+1,3:n)';
% initialise N and L
L = D; N = D;
% correct outer laplacian, N, for bounds
N(bl,bl) = N(bl,bl)/2; N(br,br) = N(br,br)/2;
N(bt,bt) = N(bt,bt)/2; N(bb,bb) = N(bb,bb)/2;
% Trick: Modify the stencil in L at the ghost points to approximate
% ddu/dndt, which gives the correct boundary conditions
L([gl;gr;gt;gb],:) = 0;
for i=gl(1:end-1)', %left
L([i,i+1],[i,i+1,i+2*n,i+2*n+1]) = ...
L([i,i+1],[i,i+1,i+2*n,i+2*n+1]) + (mu-1)/2*[1,-1,-1,1;-1,1,1,-1];
end;
for i=gr(1:end-1)', %right
L([i,i+1],[i,i+1,i-2*n,i-2*n+1]) = ...
L([i,i+1],[i,i+1,i-2*n,i-2*n+1]) + (mu-1)/2*[1,-1,-1,1;-1,1,1,-1];
end;
for i=gt(1:end-1)', %top
L([i,i+n],[i+n,i,i+n+2,i+2]) = ...
L([i,i+n],[i+n,i,i+n+2,i+2]) - (mu-1)/2*[1,-1,-1,1;-1,1,1,-1];
end;
for i=gb(1:end-1)', %bottom
L([i,i+n],[i+n,i,i+n-2,i-2]) = ...
L([i,i+n],[i+n,i,i+n-2,i-2]) - (mu-1)/2*[1,-1,-1,1;-1,1,1,-1];
end;
% make fourth order op
A = N*L;
% use bcs to eliminate ghost points
A([gl;gr;gt;gb],:) = 0;
for i=gl' %left
    A(i,[i+n,i,i+n-1,i+n+1,i+2*n]) = [2*(1+mu), -1, -mu, -mu, -1];
end
for i=gr' %right
    A(i,[i-n,i,i-n-1,i-n+1,i-2*n]) = [2*(1+mu), -1, -mu, -mu, -1];
end
for i=gt' %top
    A(i,[i+1,i,i+1+n,i+1-n,i+2]) = [2*(1+mu), -1, -mu, -mu, -1];
end
for i=gb' %bottom
    A(i,[i-1,i,i-1+n,i-1-n,i-2]) = [2*(1+mu), -1, -mu, -mu, -1];
end
% eliminate ghost points
phys = G(3:n,3:n); phys = phys(:); % put all physical nodes in a vector
ghost = [gl; gr; gt; gb];
A0 = A(phys,phys) - A(phys,ghost)/A(ghost,ghost)*A(ghost,phys);
% RHS: take into account half cells and quarter cells
B = speye(n^2);
B(bl,bl) = B(bl,bl)/2; B(br,br) = B(br,br)/2;
B(bt,bt) = B(bt,bt)/2; B(bb,bb) = B(bb,bb)/2;
B0 = B(phys,phys);
% generalised eigenvalue problem
[V,Lambda] = eigs(A0/h^4,B0,M,'SM');
[y,p] = sort(diag(Lambda));
x=[-1:2/(n-3):1];

fileID = fopen('eigenvalues.txt','w');
% plot figures
for i=4:M
    % make contour plot
    contour(x,x,reshape(V(:,p(i)),n-2,n-2));
    axis equal
    title(['Eigenvalue' num2str(Lambda(p(i),p(i))) '.'])
    % print eigenvalue
    disp(Lambda(p(i),p(i)))
    fprintf(fileID,'%12.8f\n',Lambda(p(i),p(i)));
    % EXPORTING::
    %{
    a = num2str(i);
    writematrix(reshape(V(:,p(i)),n-2,n-2),['data' a '.txt'])
    saveas(gcf,['plot' a '.png'])
    %}
end
fclose(fileID);


