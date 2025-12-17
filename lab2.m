clc
clear all

neurons=6;
cc = 10000;
lc=2;

x = 0.1:1/22:1;
x_in=0.1:1/200:1;
target = ((1+0.6*sin(2*pi*x/0.7))+0.3*sin(2*pi*x))/2;
y_out = ((1+0.6*sin(2*pi*x_in/0.7))+0.3*sin(2*pi*x_in))/2;

figure(1)
plot(x_in,y_out)
grid on


for i=1:neurons
    for k=1:lc
        w(i, k) = randn(1);
    end
    b(i, 1) = randn(1);
end
b_out = randn(1);

learningRate = 0.1;

Y=zeros(1,length(x));
v=zeros(neurons,1);
y=zeros(neurons,1);
delta=zeros(neurons,1);

for j = 1:cc
        for i = 1:length(x)
            for p = 1:neurons
                %Calculate weighted input for each neuron
                v(p, 1) = w(p, 1) * x(i) + b(p, 1);  
                y(p, 1) = tanh(v (p, 1));
            end
            
            v_out = y(1,1)*w(1,2) + y(2,1)*w(2,2) + y(3,1)*w(3,2) + y(4,1)*w(4,2) + y(5,1)*w(5,2) + y(6,1)*w(6,2) + b_out;

            %Output layer activation
            y_output = v_out;
            Y(i) = y_output;
             
            %Error calculation
            e = target(i)-y_output;

            %Hidden-layer local gradients.
            for p = 1:neurons
                delta(p, 1) = (1-tanh(v(p, 1)).^2)*e*w(p, 2);
            end
           
            %Update weights and biases for the output layer
            for p = 1:neurons
                w(p,2) = w(p,2) + learningRate * e * y(p,1);
            end
           
            % Update output bias
            b_out = b_out + learningRate * e;

            % Update weights and biases for the hidden layer
            for p = 1:neurons
                w(p, 1) = w(p, 1) + learningRate * delta(p, 1) * x(i);
                b(p, 1) = b(p, 1) + learningRate * delta(p, 1);
            end

        end
end

x_test = 0.1:1/200:1;
Y_test = zeros(size(x_test));

for i=1:length(x_test)
        v(:, 1) = w(:, 1) * x_test(i) + b(:, 1);
       
        %Activation function
        y(:, 1) = tanh(v(:, 1));
        
        %Second layer
        v_out = y(1,1)*w(1,2) + y(2,1)*w(2,2) + y(3,1)*w(3,2) + y(4,1)*w(4,2) + y(5,1)*w(5,2) + y(6,1)*w(6,2) + b_out;
        
        %Output layer activation
        y_output = v_out;
        Y_test(i) = y_output;
end

hold on
plot(x_test,Y_test)
hold off
legend('Target', 'Predicted')
