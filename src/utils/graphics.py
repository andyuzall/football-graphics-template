import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mplsoccer import Pitch


class Graphics:
    def __init__(self, width, height):
        self.width = width
        self.height = height


# Visualizar los tiros de ambos equipos con XG y nombre de jugador
def plot_event_shots(matches, shots):
    fig, ax = plt.subplots(figsize=(12, 8))
    pitch = Pitch(pitch_color="white", line_color="black")
    pitch.draw(ax=ax)
    for i in range(len(shots)):
        if shots.team_name.values[i] == matches.away_team.values[0]:
            if shots.shot_outcome_name.values[i] == "Goal":
                pitch.scatter(
                    shots.location_x.values[i],
                    shots.location_y.values[i],
                    ax=ax,
                    color="red",
                    s=shots.shot_statsbomb_xg.values[i] * 1000,
                )
                plt.text(
                    shots.location_x.values[i],
                    shots.location_y.values[i],
                    shots.player_name.values[i],
                )
            else:
                pitch.scatter(
                    shots.location_x.values[i],
                    shots.location_y.values[i],
                    ax=ax,
                    color="red",
                    s=shots.shot_statsbomb_xg.values[i] * 1000,
                    alpha=0.3,
                )
        else:
            if shots.shot_outcome_name.values[i] == "Goal":
                pitch.scatter(
                    shots.location_x.values[i],
                    80 - shots.location_y.values[i],
                    ax=ax,
                    color="blue",
                    s=shots.shot_statsbomb_xg.values[i] * 1000,
                )
                plt.text(
                    shots.location_x.values[i],
                    80 - shots.location_y.values[i],
                    shots.player_name.values[i],
                )
            else:
                pitch.scatter(
                    shots.location_x.values[i],
                    80 - shots.location_y.values[i],
                    ax=ax,
                    color="blue",
                    s=shots.shot_statsbomb_xg.values[i] * 1000,
                    alpha=0.3,
                )
    plt.text(15, 5, matches.away_team.values[0] + " shots")
    plt.text(80, 5, matches.home_team.values[0] + " shots")
    plt.title(
        "{} {} - {} Vs {}".format(
            matches.competition.values[0],
            matches.match_date.values[0],
            matches.home_team.values[0],
            matches.away_team.values[0],
        ),
        fontsize=16,
    )
    plt.show()


# Ejemplo de uso: plot_event_shots(matches, shots)


# Visualizar pases (exitosos y fallidos) por equipo
def draw_pass_event(df, team_id, fixture_uuid):
    df = df[(df['team_id'] == team_id) & (df['fixture_uuid'] == fixture_uuid)]
    mask = df['pass'] == 1

    fig, ax = plt.subplots(figsize=(15, 10))
    pitch = Pitch(pitch_color="#22312b", line_color="white")
    pitch.draw(ax=ax)

    pitch.arrows(
        df[mask].start_x,
        df[mask].start_y,
        df[mask].end_x,
        df[mask].end_y,
        color="#ad993c",
        ax=ax,
        width=2,
        headwidth=10,
        headlength=10,
        label="Pases completados",
    )
    pitch.arrows(
        df[~mask].start_x,
        df[~mask].start_y,
        df[~mask].end_x,
        df[~mask].end_y,
        color="#ba4f45",
        ax=ax,
        width=2,
        headwidth=10,
        headlength=10,
        label="Pases fallidos",
    )
    ax.legend(handlelength=5, edgecolor="white", fontsize=10, loc="lower left")
    plt.title("Pases del {}".format(team_id), fontsize=16)
    plt.show()


# Visualizar mapa de calor (pases o coordenadas de recorrido)
def plot_heat_map_pass(df, team_id, fixture_uuid, x, y, title):
    df = df[(df["team_id"] == team_id) & (df["fixture_uuid"] == fixture_uuid)]
    fig, ax = plt.subplots(figsize=(15, 10))
    pitch = Pitch(pitch_color="#22312b", line_color="white")
    pitch.draw(ax=ax)

    sns.kdeplot(
        x=df[x], y=df[y], fill=True, thresh=0.4, alpha=0.4, n_levels=10, cmap="CMRmap"
    )
    plt.title(title, fontsize=16)
    plt.show()


# Visualizar la matriz de pases por equipo
def plot_pass_matrix(df, team_id, fixture_uuid):
    df = df[(df["team_id"] == team_id) & (df["fixture_uuid"] == fixture_uuid)]
    df_pas = df[df["pass"] == 1]

    avg = df_pas.groupby("player_id").agg(
        {"start_x": ["mean", "count"], "start_y": "mean"}
    )
    avg.columns = ["x_avg", "count", "y_avg"]
    avg = avg[["x_avg", "y_avg", "count"]]

    btw = (
        df_pas.groupby(["player_id", "end_player_id"])
        .size()
        .reset_index(name="n_passes")
    )

    merg1 = btw.merge(avg, left_on="player_id", right_index=True)
    pass_btw = merg1.merge(
        avg, left_on="end_player_id", right_index=True, suffixes=["", "_end"]
    )

    pas_rec = pass_btw.groupby("end_player_id")["n_passes"].sum()
    pas_rec.name = "recibidos"
    new_avg = pd.concat([avg, pas_rec], axis=1)
    new_avg["recibidos"] = new_avg["recibidos"].fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(15, 10))
    pitch = Pitch(pitch_color="#22312b", line_color="white")
    pitch.draw(ax=ax)

    for i in range(len(pass_btw)):
        pitch.arrows(
            pass_btw.x_avg.values[i],
            pass_btw.y_avg.values[i],
            pass_btw.x_avg_end.values[i],
            pass_btw.y_avg_end.values[i],
            ax=ax,
            color="white",
            width=pass_btw.n_passes.values[i],
            headwidth=2,
            headlength=2,
            headaxislength=2,
            zorder=1,
            alpha=0.5,
        )
        pitch.scatter(
            pass_btw.x_avg.values[i],
            pass_btw.y_avg.values[i],
            s=pass_btw["count"].values[i] * 100,
            color="red",
            ax=ax,
            alpha=0.5,
        )
    for i in range(len(avg)):
        pitch.annotate(
            (new_avg["recibidos"].values[i], new_avg["count"].values[i]),
            xy=(new_avg["x_avg"].values[i], new_avg["y_avg"].values[i]),
            ax=ax,
            va="center",
            ha="center",
            color="white",
        )
    plt.show()
