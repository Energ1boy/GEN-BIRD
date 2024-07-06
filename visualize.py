import os
import neat
import visualize

def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False):
    node_names = {-1:'input1', -2: 'input2', -3: 'input3', -4: 'input4', -5: 'input5', 0:'output'} if node_names is None else node_names
    visualize.draw_net(config, genome, view=view, filename=filename, node_names=node_names, show_disabled=show_disabled, prune_unused=prune_unused)
