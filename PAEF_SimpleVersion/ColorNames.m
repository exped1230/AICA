function colornames = ColorNames(I)
%�ú�������һ��ͼ���к���black, blue, brown, gray, green, orange, pink, purple, red,
%white, yellow�Ȼ�����ɫ����������������ɫ�����ظ���
%���룺IΪһ����ɫͼ��
%�����������ɫ�����������Լ�����������ɫ�����ظ���

%��ʼ��
colornames=zeros(1,12);
[nx,ny]=size(I(:,:,1));

load('w2c.mat');

im=double(I);

% compute the color name assignment for all pixels in image im:
Color_Matrix=im2c(im,w2c,0); 

for i=1:nx
    for j=1:ny
        if Color_Matrix(i,j)==1
            colornames(2)=colornames(2)+1;
        elseif Color_Matrix(i,j)==2
            colornames(3)=colornames(3)+1;
        elseif Color_Matrix(i,j)==3
            colornames(4)=colornames(4)+1;
        elseif  Color_Matrix(i,j)==4
            colornames(5)=colornames(5)+1;
        elseif Color_Matrix(i,j)==5
            colornames(6)=colornames(6)+1;
        elseif Color_Matrix(i,j)==6
            colornames(7)=colornames(7)+1; 
        elseif Color_Matrix(i,j)==7
            colornames(8)=colornames(8)+1;
        elseif Color_Matrix(i,j)==8
            colornames(9)=colornames(9)+1;
        elseif Color_Matrix(i,j)==9
            colornames(10)=colornames(10)+1;
        elseif Color_Matrix(i,j)==10
            colornames(11)=colornames(11)+1;
        elseif Color_Matrix(i,j)==11
            colornames(12)=colornames(12)+1;
        end
    end
end
colornames(1)=sum(colornames~=0);

end

