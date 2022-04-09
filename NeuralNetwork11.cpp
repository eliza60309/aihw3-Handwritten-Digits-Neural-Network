#include <bits/stdc++.h>

#define INPUT 784
#define LAYER1 50
#define LAYER2 50
#define OUTPUT 10
#define SPEED 0.3
#define BATCH_SIZE 10
#define ITERATION 90

using namespace std;

class node
{
    public:
    double raw;
    double value;
    double bias;
    vector<double> weight;
    //vector<double> d_weight;
    
    double *function(double);
    double *dfunction(double);
};



class random_number_generator
{

    random_device dev;
    mt19937 rng;
    uniform_real_distribution<double> dist;
    public:
    random_number_generator()
    {
        rng = mt19937(dev());
        dist = uniform_real_distribution<double>(-1, 1); 
    }
    double get()
    {
        return dist(rng);
    }
};

class neural_network
{
    public:
    node input[INPUT];
    node layer1[LAYER1];
    node layer2[LAYER2];
    node output[OUTPUT];
    neural_network();
};

class back_propagation_vectors
{
    public:
    int iterations;
    vector<double> dummy;//[INPUT]
    vector<double> output_bias_error;//[OUTPUT]
    vector<double> layer2_bias_error;//[LAYER2]
    vector<double> layer1_bias_error;//[LAYER1]
    vector<vector<double> > output_weight_error;//[OUTPUT][LAYER2]
    vector<vector<double> > layer2_weight_error;//[LAYER2][LAYER1]
    vector<vector<double> > layer1_weight_error;//[LAYER1][INPUT]
    back_propagation_vectors();
    int adjust_weight(neural_network &);
    int average_weight();
    int reset_weight();
};

class entry
{
    public:
    vector<float>image;
    int value;
};

double sigmoid(double);
double dsigmoid(double);
//int softmax(node[], node[], int, int);

int set_input(node[], entry &);
int forward(neural_network &);
int backward(neural_network &, back_propagation_vectors &, double[]);

int main(int argc, char *argv[])
{
    clock_t begin, end;
    time_t begin_t, end_t;
    begin = clock();
    begin_t = time(NULL);
    //input file
    fstream data, label;
    fstream data2, label2;
    data.open(argv[1], fstream::in);
    label.open(argv[2], fstream::in);
    data2.open(argv[3], fstream::in);
    //label2.open(argv[4], fstream::in);
    vector<entry> train_data;
    vector<entry> test_data;
    vector<int> id;
    entry dummy;
    int num;
    char c;
    int cnt = 0;
    while(1)
    {
        train_data.push_back(dummy);
        for(int i = 0; i < 784; i++)
        {
            data >> num;
            if(i != 783)
                data >> c;
            train_data[cnt].image.push_back(num / 255.0f);
        }
        label >> num;
        train_data[cnt].value = num;
        id.push_back(cnt);
        if(data.fail() || label.fail())
        {
            train_data.pop_back();
            id.pop_back();
            break;
        }
        else
            cnt++;
    }
    
    cout << "train data imported: "<< cnt << endl;
    cnt = 0;
    while(1)
    {
        test_data.push_back(dummy);
        for(int i = 0; i < 784; i++)
        {
            data2 >> num;
            if(i != 783)
                data2 >> c;
            test_data[cnt].image.push_back(num / 255.0f);
        }
        //label2 >> num;
        test_data[cnt].value = num;
        if(data2.fail() || label2.fail())
        {
            test_data.pop_back();
            break;
        }
        else
            cnt++;
    }
    
    cout << "test data imported: "<< cnt << endl;


    neural_network nn;
    cout << fixed << setprecision(5);
/*
    cout << "l1:";
    for(int i = 0; i < LAYER1; i++)
        cout << nn.layer1[i].weight[300] << " ";
    cout << endl;
    cout << "l2:";
    for(int i = 0; i < LAYER2; i++)
        cout << nn.layer2[i].weight[0] << " ";
    cout << endl;
    cout << "output:";
    for(int i = 0; i < OUTPUT; i++)
        cout << nn.output[i].weight[0] << " ";
    cout << endl;*/

    int N = ITERATION;
    back_propagation_vectors bpv;
    random_number_generator random;
    vector<int> output;
    mt19937 rng;
    while(N--)
    {
        //double error = 0;
        shuffle(id.begin(), id.end(), rng);
        int j = 0;
        for(int i = 0; i < train_data.size(); i++)//i = index; j = batch index;
        {
            
            double answer[OUTPUT] = {};
            set_input(nn.input, train_data[i]);
            answer[train_data[i].value] = 1;
            forward(nn);
            backward(nn, bpv, answer);
            j++;
            if(j == BATCH_SIZE)
            {
                bpv.adjust_weight(nn);
                bpv.reset_weight();
                j = 0;
            }
            //for(int j = 0; j < OUTPUT; j++)
            //    error += abs(nn.output[j].value - answer[j]);
        }
        //cout << error << endl;
        if(j)
        {
            bpv.adjust_weight(nn);
            bpv.reset_weight();
        }
        //cout << "ITERS LEFT: " << N << endl;
        
    }
    int sum2 = 0;
    for(int i = 0; i < test_data.size(); i++)
    {
        double answer[OUTPUT] = {};
        set_input(nn.input, test_data[i]);
        //answer[test_data[i].value] = 1;
        forward(nn);
        //backward(nn, bpv, answer);
        double max = 0;
        int index = 0;
        for(int j = 0; j < OUTPUT; j++)
        {
            if(nn.output[j].value > max)
            {
                max = nn.output[j].value;
                index = j;
            }
        }
        output.push_back(index);
        //if(index == test_data[i].value)
        //    sum2++;
    }
    fstream out;
    out.open("test_predictions.csv", fstream::out);
    for(int i = 0; i < output.size(); i++)
        out << output[i] << endl;
    out.close();
    cout << "done" << endl;
    end = clock();
    end_t = time(NULL);
    int cpu_time = (end - begin) / CLOCKS_PER_SEC;
    int time_elapsed = end_t - begin_t;
    cout << "Time elapsed: " << time_elapsed / 60 << "m " << time_elapsed % 60 << "s " << endl;
    cout << "CPU time elapsed: " << cpu_time / 60 << "m " << cpu_time % 60 << "s " << endl;
    //cout << endl << "after: " << sum2 << endl;
    /*cout << "l1:";
    for(int i = 0; i < LAYER1; i++)
        cout << nn.layer1[i].weight[300] << " ";
    cout << endl;
    cout << "l2:";
    for(int i = 0; i < LAYER2; i++)
        cout << nn.layer2[i].weight[0] << " ";
    cout << endl;
    cout << "output:";
    for(int i = 0; i < OUTPUT; i++)
        cout << nn.output[i].weight[0] << " ";
    cout << endl;*/
    return 0;
}

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}
                    
double dsigmoid(double x)
{
    return sigmoid(x) * (1.0 - sigmoid(x));
}

int set_input(node input[], entry &data)
{
    for(int i = 0; i < INPUT; i++)
        input[i].value = data.image[i];
}

int forward(neural_network &nn) //fwd feeding
{
    node *input = nn.input;
    node *layer1 = nn.layer1;
    node *layer2 = nn.layer2;
    node *output = nn.output;
    for(int i = 0; i < LAYER1; i++)
    {
        layer1[i].raw = 0;
        for(int j = 0; j < INPUT; j++)
            layer1[i].raw += input[j].value * layer1[i].weight[j];
        layer1[i].raw += layer1[i].bias;
        layer1[i].value = sigmoid(layer1[i].raw);
    }
    for(int i = 0; i < LAYER2; i++)
    {
        layer2[i].raw = 0;
        for(int j = 0; j < LAYER1; j++)
            layer2[i].raw += layer1[j].value * layer2[i].weight[j];
        layer2[i].raw += layer2[i].bias;
        layer2[i].value = sigmoid(layer2[i].raw);
    }
    for(int i = 0; i < OUTPUT; i++)
    {
        output[i].raw = 0;
        for(int j = 0; j < LAYER2; j++)
            output[i].raw += layer2[j].value * output[i].weight[j];
        output[i].raw += output[i].bias;
        output[i].value = sigmoid(output[i].raw);
    }
}

int backward(neural_network &nn, back_propagation_vectors &bpv, double answer[])//back ppgation //sums all error up until resets
{
    node *input = nn.input;
    node *layer1 = nn.layer1;
    node *layer2 = nn.layer2;
    node *output = nn.output;
    /*vector<double> *dummy = bpv.dummy;//[INPUT]
    vector<double> *output_bias_error = bpv.output_bias_error;//[OUTPUT]
    vector<double> *layer2_bias_error = bpv.layer2_bias_error;//[LAYER2]
    vector<double> *layer1_bias_error = bpv.layer1_bias_error;//[LAYER1]
    vector<vector<double> > *output_weight_error = bpv.output_weight_error;//[OUTPUT][LAYER2]
    vector<vector<double> > *layer2_weight_error = bpv.layer2_weight_error;//[LAYER2][LAYER1]
    vector<vector<double> > *layer1_weight_error = bpv.layer1_weight_error;//[LAYER1][INPUT]*/
    
    //local vectors resize
    vector<double> output_accum;//[OUTPUT]
    vector<double> layer2_accum;//[LAYER2]
    vector<double> layer1_accum;//[LAYER1]
    output_accum.resize(OUTPUT, 0);
    layer2_accum.resize(LAYER2, 0);
    layer1_accum.resize(LAYER1, 0);
    for(int i = 0; i < OUTPUT; i++)
    {
        output_accum[i] = (output[i].value - answer[i]) * dsigmoid(output[i].raw);
        bpv.output_bias_error[i] += output_accum[i];
        //output_weight_error.push_back(vector<double>());//[OUTPUT][LAYER2]
        for(int j = 0; j < LAYER2; j++)
            bpv.output_weight_error[i][j] += output_accum[i] * layer2[j].value;
    }
    for(int i = 0; i < LAYER2; i++)
    {
        //layer2_accum.push_back(0);
        for(int j = 0; j < OUTPUT; j++)
            layer2_accum[i] += output[j].weight[i] * dsigmoid(layer2[i].raw) * output_accum[j];
        bpv.layer2_bias_error[i] += layer2_accum[i];
        //layer2_weight_error.push_back(vector<double>());//[LAYER2][LAYER1]
        for(int j = 0; j < LAYER1; j++)
            bpv.layer2_weight_error[i][j] += layer2_accum[i] * layer1[j].value;
    }
    //cout << "mark" << layer2[10].raw << endl;
    for(int i = 0; i < LAYER1; i++)
    {
        //layer1_accum.push_back(0);
        for(int j = 0; j < LAYER2; j++)
            layer1_accum[i] += layer2[j].weight[i] * dsigmoid(layer1[i].raw) * layer2_accum[j];
        bpv.layer1_bias_error[i] += layer1_accum[i];
        //layer1_weight_error.push_back(vector<double>());//[LAYER1][INPUT]
        for(int j = 0; j < INPUT; j++)
            bpv.layer1_weight_error[i][j] += layer1_accum[i] * input[j].value;
    }
    bpv.iterations++;
}
neural_network::neural_network()
{
    
    random_number_generator random;
    //assign random weight n bias
    for(int i = 0; i < LAYER1; i++)
    {
        for(int j = 0; j < INPUT; j++)
            layer1[i].weight.push_back(random.get());
        layer1[i].bias = random.get();
    }
    for(int i = 0; i < LAYER2; i++)
    {
        for(int j = 0; j < LAYER1; j++)
            layer2[i].weight.push_back(random.get());
        layer2[i].bias = random.get();
    }
    for(int i = 0; i < OUTPUT; i++)
    {
        for(int j = 0; j < LAYER2; j++)
            output[i].weight.push_back(random.get());
        output[i].bias = random.get();
    }
}

back_propagation_vectors::back_propagation_vectors()
{
    iterations = 0;
    output_bias_error.resize(OUTPUT, 0);
    layer2_bias_error.resize(LAYER2, 0);
    layer1_bias_error.resize(LAYER1, 0);
    dummy.resize(INPUT, 0);
    output_weight_error.resize(OUTPUT, layer2_bias_error);
    layer2_weight_error.resize(LAYER2, layer1_bias_error);
    layer1_weight_error.resize(LAYER1, dummy);
}

int back_propagation_vectors::adjust_weight(neural_network &nn)
{
    if(!iterations)
        return 0;
    node *input = nn.input;
    node *layer1 = nn.layer1;
    node *layer2 = nn.layer2;
    node *output = nn.output;
    for(int i = 0; i < LAYER1; i++)
    {
        for(int j = 0; j < INPUT; j++)
            layer1[i].weight[j] -= layer1_weight_error[i][j] * SPEED / iterations;
        layer1[i].bias -= layer1_bias_error[i] * SPEED / iterations;
    }

    for(int i = 0; i < LAYER2; i++)
    {
        for(int j = 0; j < LAYER1; j++)
            layer2[i].weight[j] -= layer2_weight_error[i][j] * SPEED / iterations;
        layer2[i].bias -= layer2_bias_error[i] * SPEED / iterations;

    }

    for(int i = 0; i < OUTPUT; i++)
    {
        for(int j = 0; j < LAYER2; j++)
            output[i].weight[j] -= output_weight_error[i][j] * SPEED / iterations;
        output[i].bias -= output_bias_error[i] * SPEED / iterations;
    }
}

int back_propagation_vectors::reset_weight()
{
    iterations = 0;
    for(int i = 0; i < LAYER1; i++)
    {
        for(int j = 0; j < INPUT; j++)
            layer1_weight_error[i][j] = 0;
        layer1_bias_error[i] = 0;
    }

    for(int i = 0; i < LAYER2; i++)
    {
        for(int j = 0; j < LAYER1; j++)
            layer2_weight_error[i][j] = 0;
        layer2_bias_error[i] = 0;

    }

    for(int i = 0; i < OUTPUT; i++)
    {
        for(int j = 0; j < LAYER2; j++)
            output_weight_error[i][j] = 0;
        output_bias_error[i] = 0;
    }
}

/*
int softmax(node input[], node output[], int in_size, int out_size) 
{
    vector<double> exparray;
    for(int i = 0; i < out_size; i++)
    {
        double sum = 0;
        for(int j = 0; j < in_size; j++)
            sum += input[j].value * output[i].weight[j];
        exparray.push_back(exp(sum));
    }
    double expsum = 0;
    for(int i = 0; i < out_size; i++)
        expsum += exparray[i];
    for(int i = 0; i < out_size; i++)
        output[i].value = exparray[i] / expsum;
    return 0;
}
*/
//relu? 
