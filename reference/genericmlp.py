class mlpQNetwork(baseQNetwork):
    def __init__(self, env, paramdict):
        super(mlpQNetwork, self).__init__(env)
        
        self.input_size = self.env.observation_space.shape[0]
        self.output_size = self.env.action_space.n
        self.hidden1 = paramdict['hidden1']
        self.hidden2 = paramdict['hidden2']
        self.hidden3 = paramdict['hidden3']
        
        self.linear1 = nn.Linear(self.input_size, self.hidden1)
        if self.hidden2 is None:
            self.linear2 = nn.Linear(self.hidden1, self.output_size)
        else:
            self.linear2 = nn.Linear(self.hidden1, self.hidden2)
            if self.hidden3 is None:
                self.linear3 = nn.Linear(self.hidden2, self.output_size)
            else:
                self.linear3 = nn.Linear(self.hidden2, self.hidden3)
                self.linear4 = nn.Linear(self.hidden3, self.output_size)

    def forward(self, input_state):
        hidden1 = nn.functional.relu(self.linear1(input_state))
        if self.hidden2 is None:
            output = nn.functional.relu(self.linear2(hidden1))
        else:
            hidden2 = nn.functional.relu(self.linear2(hidden1))
            if self.hidden3 is None:
                output = nn.functional.relu(self.linear3(hidden2))
            else:
                hidden3 = nn.functional.relu(self.linear3(hidden2))
                output = nn.functional.relu(self.linear4(hidden3))
        return output
