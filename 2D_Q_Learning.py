import numpy as np
import random
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
import threading

# --- 1. Definition de l'environnement et Parametres ---

TAILLE_GRILLE = 20
ACTIONS = {0: 'HAUT', 1: 'BAS', 2: 'GAUCHE', 3: 'DROITE'}
NB_ACTIONS = len(ACTIONS)

RECOMPENSE_CIBLE = 10
RECOMPENSE_DEPLACEMENT = -1
RECOMPENSE_COLLISION = -10
EPISODES = 5000

# --- Fonctions de l'Environnement (Identiques a votre code) ---

def initialiser_grille_positions():
    
    positions_possibles = list(range(TAILLE_GRILLE * TAILLE_GRILLE))
    
    nb_elements_fixes = 2 
    nb_obstacles_aleatoires = 100
    
    # Assurez-vous que le nombre d'éléments à choisir n'est pas supérieur à la taille de la grille
    nb_total_a_choisir = min(TAILLE_GRILLE * TAILLE_GRILLE, nb_elements_fixes + nb_obstacles_aleatoires)
    positions_choisies = random.sample(positions_possibles, nb_total_a_choisir)
    
    pos_depart = positions_choisies[0]
    pos_cible = positions_choisies[1]
    
    pos_obstacles = positions_choisies[2:]
    
    def to_coords(pos):
        return (pos // TAILLE_GRILLE, pos % TAILLE_GRILLE)

    coords_depart = to_coords(pos_depart)
    coords_cible = to_coords(pos_cible)
    coords_obstacles = [to_coords(p) for p in pos_obstacles]

    return coords_depart, coords_cible, coords_obstacles

def obtenir_etat_recompense_suivants(etat_courant, action_index, coords_cible, coords_obstacles):
    r, c = etat_courant
    new_r, new_c = r, c

    if action_index == 0: new_r -= 1
    elif action_index == 1: new_r += 1
    elif action_index == 2: new_c -= 1
    elif action_index == 3: new_c += 1

    nouvel_etat = (new_r, new_c)

    if not (0 <= new_r < TAILLE_GRILLE and 0 <= new_c < TAILLE_GRILLE):
        return etat_courant, RECOMPENSE_COLLISION, True 
    if nouvel_etat in coords_obstacles:
        return etat_courant, RECOMPENSE_COLLISION, True 
    if nouvel_etat == coords_cible:
        return nouvel_etat, RECOMPENSE_CIBLE, True 
    
    return nouvel_etat, RECOMPENSE_DEPLACEMENT, False 

# --- 2. Fonction d'entrainement (Modifiee pour l'anti-boucle) ---

class QLearningAgent:
    def __init__(self, episodes):
        self.EPISODES = episodes
        self.coords_depart, self.coords_cible, self.coords_obstacles = initialiser_grille_positions()
        self.Q_table = np.zeros((TAILLE_GRILLE * TAILLE_GRILLE, NB_ACTIONS))
        self.historique_recompenses = []
        self.meilleure_recompense = -float('inf')
        self.meilleur_chemin = []
        
        self.ALPHA = 0.1
        self.GAMMA = 0.9
        self.EPSILON = 1
        self.DECAY_EPSILON = 0.9995
        self.MIN_EPSILON = 0.01

        # *** AJUSTEMENT ANTI-BOUCLE (Calcul de la limite de pas theorique) ***
        nombre_etats_libres = TAILLE_GRILLE * TAILLE_GRILLE - len(self.coords_obstacles)
        facteur_allongement = 2.5 # Facteur pour autoriser les détours (peut être ajusté)
        
        # Le chemin le plus long théorique sans boucle est (nombre_etats_libres - 1)
        # On multiplie par un facteur pour gérer les détours autour des obstacles.
        self.MAX_STEPS_EPISODE = int(nombre_etats_libres * facteur_allongement)
        
        # S'assurer d'une limite minimum raisonnable même dans une petite grille
        if self.MAX_STEPS_EPISODE < TAILLE_GRILLE * 2: 
             self.MAX_STEPS_EPISODE = TAILLE_GRILLE * 2
        # Fin de l'ajustement
        

    def train(self, progress_callback=None):
        self.historique_recompenses = []
        self.Q_table = np.zeros((TAILLE_GRILLE * TAILLE_GRILLE, NB_ACTIONS))
        self.EPSILON = 1.0
        self.meilleure_recompense = -float('inf')
        
        for episode in range(self.EPISODES):
            etat_coords_courant = self.coords_depart
            etat_idx_courant = etat_coords_courant[0] * TAILLE_GRILLE + etat_coords_courant[1]
            termine = False
            chemin_episode = [etat_coords_courant]
            recompense_episode = 0
            
            while not termine:
                
                if random.uniform(0, 1) < self.EPSILON:
                    action_index = random.choice(list(ACTIONS.keys()))
                else:
                    max_q_value = np.max(self.Q_table[etat_idx_courant, :])
                    meilleures_actions = np.where(self.Q_table[etat_idx_courant, :] == max_q_value)[0]
                    action_index = random.choice(meilleures_actions)
                    
                etat_coords_suivant, recompense, termine_env = obtenir_etat_recompense_suivants(
                    etat_coords_courant, action_index, self.coords_cible, self.coords_obstacles
                )
                
                etat_idx_suivant = etat_coords_suivant[0] * TAILLE_GRILLE + etat_coords_suivant[1]
                
                recompense_episode += recompense
                chemin_episode.append(etat_coords_suivant)
                
                # Mise a jour de la Q-table
                q_value_ancien = self.Q_table[etat_idx_courant, action_index]
                max_q_suivant = np.max(self.Q_table[etat_idx_suivant, :]) if not termine_env else 0
                q_value_nouveau = (1 - self.ALPHA) * q_value_ancien + self.ALPHA * (recompense + self.GAMMA * max_q_suivant)
                self.Q_table[etat_idx_courant, action_index] = q_value_nouveau
                
                etat_coords_courant = etat_coords_suivant
                etat_idx_courant = etat_idx_suivant
                
                # *** AJUSTEMENT ANTI-BOUCLE (Condition d'arrêt prématuré) ***
                if len(chemin_episode) > self.MAX_STEPS_EPISODE:
                    # L'agent est dans une boucle ou prend un chemin trop long
                    termine = True
                    # On peut ajouter une pénalité sévère si la cible n'est pas atteinte
                    if etat_coords_courant != self.coords_cible:
                        recompense_episode += RECOMPENSE_COLLISION * 2 # Pénalité pour échec
                else:
                    termine = termine_env
                # Fin de l'ajustement
                    
            if self.EPSILON > self.MIN_EPSILON:
                self.EPSILON *= self.DECAY_EPSILON

            self.historique_recompenses.append(recompense_episode)
            
            if recompense_episode > self.meilleure_recompense and etat_coords_courant == self.coords_cible:
                self.meilleure_recompense = recompense_episode
                self.meilleur_chemin = list(chemin_episode)
            
            if progress_callback:
                progress_callback(episode + 1, self.meilleure_recompense)
                
        return self.Q_table, self.meilleur_chemin, self.meilleure_recompense, self.historique_recompenses

# --- 3. Fonction pour extraire le chemin optimal (Ajustee pour l'anti-boucle) ---

def obtenir_chemin_optimal(Q_table, coords_depart, coords_cible, coords_obstacles):
    
    # *** AJUSTEMENT ANTI-BOUCLE (Recalcul de la limite pour l'extraction) ***
    nombre_etats_libres = TAILLE_GRILLE * TAILLE_GRILLE - len(coords_obstacles)
    facteur_allongement = 2.5 
    max_etapes = int(nombre_etats_libres * facteur_allongement)
    if max_etapes < TAILLE_GRILLE * 2: 
        max_etapes = TAILLE_GRILLE * 2
    # Fin de l'ajustement

    etat_coords_courant = coords_depart
    chemin = [etat_coords_courant]
    
    etapes = 0
    
    while etat_coords_courant != coords_cible and etapes < max_etapes:
        r, c = etat_coords_courant
        etat_idx_courant = r * TAILLE_GRILLE + c
        
        # Choix de l'action déterministe (meilleure Q-value)
        max_q_value = np.max(Q_table[etat_idx_courant, :])
        meilleures_actions = np.where(Q_table[etat_idx_courant, :] == max_q_value)[0]
        action_index = random.choice(meilleures_actions)
        
        etat_coords_suivant, _, _ = obtenir_etat_recompense_suivants(
            etat_coords_courant, action_index, coords_cible, coords_obstacles
        )
        
        # Mécanisme pour briser les boucles déterministes (si le meilleur Q-value
        # ramène à la même case sans être la cible, il y a une boucle)
        if etat_coords_suivant == etat_coords_courant and etat_coords_suivant != coords_cible:
            break

        chemin.append(etat_coords_suivant)
        etat_coords_courant = etat_coords_suivant
        etapes += 1
        
    return chemin

# --- 4. Interface Graphique (GUI) avec Tkinter et Matplotlib ---

class QLearningGUI:
    def __init__(self, master, agent):
        self.master = master
        self.agent = agent
        master.title("Q-Learning Agent - Grille 20x20")
        
        self.current_path = []
        self.current_step = 0
        self.replay_running = False

        # Configuration de l'interface
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(pady=10, padx=10, fill="both", expand=True)

        # Onglet 1: Grille et Replay
        self.frame_grid = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_grid, text="Grille & Replay")
        self.setup_grid_frame(self.frame_grid)

        # Onglet 2: Convergence
        self.frame_chart = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_chart, text="Historique d'Apprentissage")
        self.setup_chart_frame(self.frame_chart)

        # Démarrage de l'entraînement dans un thread séparé
        threading.Thread(target=self.start_training).start()

    def setup_grid_frame(self, frame):
        # Frame pour la grille
        self.canvas_grid = tk.Canvas(frame, width=40 * TAILLE_GRILLE, height=40 * TAILLE_GRILLE, bg="white")
        self.canvas_grid.pack(side=tk.LEFT, padx=10, pady=10)

        # Frame pour les contrôles et infos
        frame_controls = ttk.Frame(frame)
        frame_controls.pack(side=tk.RIGHT, padx=10, pady=10, fill="y")
        
        # État et Controles
        ttk.Label(frame_controls, text="PROGRESSION", font=('Arial', 12, 'bold')).pack(pady=5)
        self.progress_bar = ttk.Progressbar(frame_controls, orient='horizontal', length=200, mode='determinate', maximum=self.agent.EPISODES)
        self.progress_bar.pack(pady=5)
        self.label_episode = ttk.Label(frame_controls, text="Episode: 0 / 5000")
        self.label_episode.pack(pady=2)
        self.label_best_reward = ttk.Label(frame_controls, text="Meilleure Récompense: N/A")
        self.label_best_reward.pack(pady=2)
        
        ttk.Separator(frame_controls, orient='horizontal').pack(fill='x', pady=10)

        ttk.Label(frame_controls, text="REPLAY", font=('Arial', 12, 'bold')).pack(pady=5)
        
        self.button_replay = ttk.Button(frame_controls, text="Lancer Replay Optimal", command=self.start_replay, state=tk.DISABLED)
        self.button_replay.pack(pady=5)
        
        self.label_replay_status = ttk.Label(frame_controls, text="Statut: En attente d'entrainement...")
        self.label_replay_status.pack(pady=2)
        
        self.draw_grid()

    def setup_chart_frame(self, frame):
        # La taille de la grille est TAILLE_GRILLE=20, ajustons la taille de la figure si besoin
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas_chart = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas_widget = self.canvas_chart.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        self.plot_convergence() # Initial plot

    def draw_grid(self, path=None):
        self.canvas_grid.delete("all")
        # Ajustons la taille des cellules pour une grille 20x20. 20 * 20 = 400.
        # Taille du canvas est 40 * 20 = 800. Donc cell_size = 40. C'est OK.
        cell_size = 40 
        
        # Dessiner la grille
        for r in range(TAILLE_GRILLE):
            for c in range(TAILLE_GRILLE):
                x1, y1 = c * cell_size, r * cell_size
                x2, y2 = x1 + cell_size, y1 + cell_size
                
                color = "lightgray"
                content = ""
                
                coords = (r, c)
                
                if coords == self.agent.coords_cible:
                    color = "green"
                    content = "T" # Cible
                elif coords in self.agent.coords_obstacles:
                    color = "black"
                    content = "X" # Obstacle

                self.canvas_grid.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")
                if content:
                     self.canvas_grid.create_text(x1 + cell_size/2, y1 + cell_size/2, text=content, fill="white" if color in ["black", "green"] else "black", font=('Arial', 8, 'bold'))

        # Dessiner le chemin (si spécifié)
        if path:
            for i, (r, c) in enumerate(path):
                x_center = c * cell_size + cell_size / 2
                y_center = r * cell_size + cell_size / 2
                
                # Trace du chemin
                if i > 0:
                    prev_r, prev_c = path[i-1]
                    prev_x = prev_c * cell_size + cell_size / 2
                    prev_y = prev_r * cell_size + cell_size / 2
                    self.canvas_grid.create_line(prev_x, prev_y, x_center, y_center, fill="blue", width=2, arrow=tk.LAST)
                
                # Nœuds du chemin
                color = "blue"
                if (r, c) == self.agent.coords_cible: color = "orange"
                if (r, c) == self.agent.coords_depart and i == 0: color = "cyan"

                self.canvas_grid.create_oval(x_center - 5, y_center - 5, x_center + 5, y_center + 5, fill=color, outline=color)

        # Dessiner l'Agent (dernière position du chemin ou position de départ)
        agent_r, agent_c = path[-1] if path else self.agent.coords_depart
        x_agent = agent_c * cell_size + cell_size / 2
        y_agent = agent_r * cell_size + cell_size / 2
        self.canvas_grid.create_oval(x_agent - 15, y_agent - 15, x_agent + 15, y_agent + 15, fill="red", outline="red")
        self.canvas_grid.create_text(x_agent, y_agent, text="A", fill="white", font=('Arial', 10, 'bold'))


    def update_training_status(self, episode, best_reward):
        # Fonction appelée par le thread d'entraînement
        self.progress_bar['value'] = episode
        self.label_episode['text'] = f"Episode: {episode} / {self.agent.EPISODES}"
        self.label_best_reward['text'] = f"Meilleure Récompense: {best_reward:.2f}"
        
        # Mise à jour de l'affichage de la grille (montre la position de l'agent au depart)
        if episode == 1:
            self.draw_grid()

    def training_complete(self):
        # Fonction appelée une fois l'entraînement terminé
        messagebox.showinfo("Entrainement Terminé", f"Le Q-learning est terminé après {self.agent.EPISODES} épisodes.\nMeilleure Récompense: {self.agent.meilleure_recompense:.2f}")
        
        # Calcul du chemin optimal final
        Q_table = self.agent.Q_table
        self.final_optimal_path = obtenir_chemin_optimal(Q_table, self.agent.coords_depart, self.agent.coords_cible, self.agent.coords_obstacles)
        
        # Affichage des résultats
        self.draw_grid(self.final_optimal_path)
        self.plot_convergence()
        
        self.button_replay.config(state=tk.NORMAL)
        self.label_replay_status['text'] = f"Statut: Prêt (Longueur: {len(self.final_optimal_path) - 1} étapes)"


    def start_training(self):
        Q_table, meilleur_chemin, meilleure_recompense, historique_recompenses = self.agent.train(self.update_training_status)
        self.master.after(0, self.training_complete) # Exécuter la fin de l'entraînement dans le thread principal

    def plot_convergence(self):
        if not self.agent.historique_recompenses:
            self.ax.clear()
            self.ax.text(0.5, 0.5, "En attente des donnees d'entrainement...", ha='center', va='center', fontsize=12)
            self.ax.set_title("Historique d'Apprentissage Q-Learning")
            self.canvas_chart.draw()
            return

        historique_recompenses = self.agent.historique_recompenses
        fenetre_lissage = 100
        recompenses_moyenne_glissante = np.convolve(historique_recompenses, np.ones(fenetre_lissage)/fenetre_lissage, mode='valid')
        
        self.ax.clear()
        
        # Courbe 1: Recompense par episode (valeurs brutes)
        self.ax.plot(historique_recompenses, alpha=0.3, color='gray', label="Recompense par Episode (Brut)")
        
        # Courbe 2: Moyenne Glissante
        self.ax.plot(np.arange(len(recompenses_moyenne_glissante)) + fenetre_lissage - 1, 
                     recompenses_moyenne_glissante, 
                     color='red', 
                     label=f"Moyenne Glissante (Fenetre {fenetre_lissage})")
        
        self.ax.set_title("Convergence de l'apprentissage Q-Learning")
        self.ax.set_xlabel("Numero d'Episode")
        self.ax.set_ylabel("Recompense Cumulee")
        self.ax.legend()
        self.ax.grid(True, linestyle='--', alpha=0.6)
        
        self.canvas_chart.draw()


    def start_replay(self):
        if self.replay_running or not self.final_optimal_path:
            return
        
        self.current_path = self.final_optimal_path
        self.current_step = 0
        self.replay_running = True
        self.label_replay_status['text'] = "Statut: Replay en cours..."
        self.button_replay.config(text="Replay en cours...", state=tk.DISABLED)
        self.replay_step()

    def replay_step(self):
        if self.current_step < len(self.current_path):
            current_sub_path = self.current_path[:self.current_step + 1]
            self.draw_grid(current_sub_path)
            
            self.label_replay_status['text'] = f"Statut: Étape {self.current_step} / {len(self.current_path) - 1}"
            
            self.current_step += 1
            # Replay Vitesse (200 ms par étape)
            self.master.after(200, self.replay_step) 
        else:
            self.replay_running = False
            self.label_replay_status['text'] = "Statut: Replay terminé."
            self.button_replay.config(text="Rejouer Optimal", state=tk.NORMAL)
            
# --- 5. Lancement de l'Application ---

if __name__ == '__main__':
    root = tk.Tk()
    
    # Redefinition de la taille de la grille pour la demonstration GUI

    EPISODES = 5000
    
    agent = QLearningAgent(episodes=EPISODES)
    app = QLearningGUI(root, agent)
    
    root.mainloop()