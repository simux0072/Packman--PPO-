class bar_update():
    def __init__(self):
        self.bar_lenght = 50
        self.bar_prog = 0
        self.bar = list('[' + ' ' * (self.bar_lenght) + ']')

    def new_bar(self):
        self.bar = list('[' + ' ' * (self.bar_lenght) + ']')
        self.bar_prog = 0

    def print_info(self, num_ep, loss, score, high_school, update, num_update):
        base = int(num_ep / (update / self.bar_lenght)) 
        if base != self.bar_prog:
            self.bar_prog = base
            if base == 1:
                self.bar[1] = '>'
            else:
                self.bar[base - 1] = '='
                self.bar[base] = '>'
        print(''.join(self.bar) + ' Episode: ' + str(num_update) + ' Game: ' + str(num_ep) + ' loss: ' + str(loss) + ' score: ' + str(score) + ' High Score: ' + str(high_school) + ' ' * 20, end='\r')