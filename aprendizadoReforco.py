import gym
import random
import numpy as np
from IPython.display import clear_output

env = gym.make('Taxi-v3')

alpha = 0.1
gamma = 0.6
epsilon = 0.1

q_table = np.zeros([env.observation_space.n, env.action_space.n])

for i in range(1000000):
    estado = env.reset()
    penalidades, recompensa = 0, 0
    done = False

    while not done:
        # Exploração
        if random.uniform(0, 1) < epsilon:
            acao = env.action_space.sample()
        # Exploitation
        else:
            acao = np.argmax(q_table[estado])

        proximo_estado, recompensa, done, info = env.step(acao)

        q_antigo = q_table[estado, acao]
        proximo_maximo = np.max(q_table[proximo_estado])

        q_novo = (1 - alpha) * q_antigo + alpha * \
                  (recompensa + gamma * proximo_maximo)

        q_table[estado, acao] = q_novo

        if recompensa == -10:
            penalidades += 1

        estado = proximo_estado

    if i % 100 == 0:
        clear_output(wait=True)
        print('Episódio: ', i)

print('Treinamento concluído')

total_penalidade = 0
episodios = 50
frames = []

for _ in range(episodios):
  estado = env.reset()
  penalidades, recompensa = 0, 0
  done = False
  while not done:
    acao = np.argmax(q_table[estado])
    estado, recompensa, done, info = env.step(acao)

    if recompensa == -10:
      penalidades += 1

    frames.append({
        'frame': env.render(mode='ansi'),
        'state': estado,
        'action': acao,
        'reward': recompensa
    })

    total_penalidade += penalidades


print('Episódios', episodios)
print('Penalidades', total_penalidades)


from time import sleep
for frame in frames:
  clear_output(wait=True)
  print(frame['frame'])
  print('Estado', frame['state'])
  print('Ação', frame['action'])
  print('Recompensa', frame['reward'])
  sleep(.5)


  