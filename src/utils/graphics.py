import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
import seaborn as sns
from mplsoccer import Pitch, VerticalPitch


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


# Visualizar pases (exitosos y fallidos) por jugador usando grid (un campo por player_name)
def draw_pass_event(df, team_id, fixture_uuid, ncols=3, figheight=20, player_col=None):
    """
    Dibuja un campo por jugador con sus pases (completados y fallidos) usando pitch.grid.

    Args:
        df: DataFrame con columnas team_id, fixture_uuid, pass, start_x, start_y, end_x, end_y
            y player_name (o player_id si se pasa player_col).
        team_id: ID del equipo.
        fixture_uuid: UUID del partido.
        ncols: Número de columnas del grid de campos.
        figheight: Altura total de la figura en pulgadas.
        player_col: Columna para agrupar por jugador ('player_name' o 'player_id'). Si None, usa 'player_name' si existe, sino 'player_id'.
    """
    df = df[(df["team_id"] == team_id) & (df["fixture_uuid"] == fixture_uuid)].copy()
    if df.empty:
        raise ValueError("No hay eventos para ese team_id y fixture_uuid.")

    if player_col is None:
        player_col = "player_name" if "player_name" in df.columns else "player_id"
    if player_col not in df.columns:
        raise ValueError(f"El DataFrame no tiene la columna '{player_col}'.")

    # Solo filas que son pases (tienen start_x, end_x, etc.)
    df = df.dropna(subset=["start_x", "start_y", "end_x", "end_y"])
    if df.empty:
        raise ValueError("No hay pases con coordenadas para ese equipo y partido.")

    players = df[player_col].dropna().unique().tolist()
    n_players = len(players)
    if n_players == 0:
        raise ValueError("No hay jugadores con pases.")

    nrows = int(np.ceil(n_players / ncols))
    pitch = Pitch(pitch_color="#22312b", line_color="white")

    fig, axs = pitch.grid(
        nrows=nrows,
        ncols=ncols,
        figheight=figheight,
        grid_height=0.84,
        grid_width=0.95,
        space=0.05,
        bottom=0.025,
        endnote_height=0.03,
        endnote_space=0.01,
        title_height=0.06,
        title_space=0.01,
        axis=True,
    )

    for idx, ax in enumerate(axs["pitch"].flat):
        if idx >= n_players:
            ax.set_visible(False)
            continue
        player = players[idx]
        df_player = df[df[player_col] == player]
        mask = df_player["pass"] == 1
        complete = df_player[mask]
        failed = df_player[~mask]

        if not complete.empty:
            pitch.arrows(
                complete.start_x,
                complete.start_y,
                complete.end_x,
                complete.end_y,
                color="#ad993c",
                ax=ax,
                width=2,
                headwidth=10,
                headlength=10,
            )
        if not failed.empty:
            pitch.arrows(
                failed.start_x,
                failed.start_y,
                failed.end_x,
                failed.end_y,
                color="#ba4f45",
                ax=ax,
                width=2,
                headwidth=10,
                headlength=10,
            )

        total = len(complete) + len(failed)
        pct = round(100 * len(complete) / total, 1) if total else 0
        ax.set_title(
            f"{player} | {len(complete)}/{total} ({pct}%)",
            fontsize=10,
            color="white",
        )

    if "title" in axs:
        axs["title"].text(
            0.5,
            0.5,
            f"Pases por jugador (team_id={team_id})",
            fontsize=14,
            color="white",
            ha="center",
            va="center",
        )
    if "endnote" in axs:
        axs["endnote"].text(
            0.5,
            0.5,
            "Amarillo: completados | Rojo: fallidos",
            fontsize=10,
            color="white",
            ha="center",
            va="center",
        )

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

    # Estilo: campo verde, líneas blancas, círculos/pases azul claro, texto blanco sans-serif
    fig, ax = plt.subplots(figsize=(15, 10))
    fig.patch.set_facecolor("#2d5016")
    pitch = Pitch(pitch_color="grass", line_color="white", stripe=True, stripe_color="#2d5016")
    pitch.draw(ax=ax)

    # Azul claro/celeste para pases (con transparencia) y círculos; borde azul oscuro
    color_pases = "#7dd3fc"
    color_borde = "#1e3a5f"

    for i in range(len(pass_btw)):
        pitch.arrows(
            pass_btw.x_avg.values[i],
            pass_btw.y_avg.values[i],
            pass_btw.x_avg_end.values[i],
            pass_btw.y_avg_end.values[i],
            ax=ax,
            color=color_pases,
            width=pass_btw.n_passes.values[i],
            headwidth=2,
            headlength=2,
            headaxislength=2,
            zorder=1,
            alpha=0.65,
        )
        pitch.scatter(
            pass_btw.x_avg.values[i],
            pass_btw.y_avg.values[i],
            s=pass_btw["count"].values[i] * 100,
            color=color_pases,
            edgecolors=color_borde,
            linewidths=1.5,
            ax=ax,
            alpha=0.9,
            zorder=2,
        )
    for i in range(len(avg)):
        pitch.annotate(
            (new_avg["recibidos"].values[i], new_avg["count"].values[i]),
            xy=(new_avg["x_avg"].values[i], new_avg["y_avg"].values[i]),
            ax=ax,
            va="center",
            ha="center",
            color="white",
            fontsize=10,
            fontweight="bold",
            fontfamily="sans-serif",
        )
    plt.show()

# Visualizar heatmap de acciones a balón parado por zonas
def plot_heatmap_set_pieces(
    df,
    x_col="start_x",
    y_col="start_y",
    team_name="",
    name="",
):
    df = df[(df['free_kick_pass'] == 1) | (df['free_kick_shot'] == 1) | (df['corner_kick'] == 1)]
    df_set = df[(df['team_name'] == team_name) & (df['name'] == name)]
    if df_set.empty:
        raise ValueError("No hay filas con acciones a balón parado (revisa set_piece_columns).")

    pitch = VerticalPitch(
        pitch_type="skillcorner", 
        line_zorder=2,
        pitch_width=68,
        pitch_length=105,
        pitch_color="#f4edf0"
    )
    fig, ax = pitch.draw(figsize=(4.125, 6))
    fig.set_facecolor("#f4edf0")

    bin_x = np.linspace(pitch.dim.left, pitch.dim.right, num=7)
    bin_y = np.sort(
        np.array(
            [
                pitch.dim.bottom,
                pitch.dim.six_yard_bottom,
                pitch.dim.six_yard_top,
                pitch.dim.top,
            ]
        )
    )
    bin_statistic = pitch.bin_statistic(
        df_set[x_col].values,
        df_set[y_col].values,
        statistic="count",
        bins=(bin_x, bin_y),
        normalize=True,
    )
    pitch.heatmap(
        bin_statistic, ax=ax, cmap="Reds", edgecolor="#f9f9f9"
    )
    path_eff = [
        path_effects.Stroke(linewidth=1.5, foreground="black"),
        path_effects.Normal(),
    ]
    pitch.label_heatmap(
        bin_statistic,
        color="#f4edf0",
        fontsize=18,
        ax=ax,
        ha="center",
        va="center",
        str_format="{:.0%}",
        path_effects=path_eff,
    )
    plt.title("ABP | {} - {}".format(team_name, name), fontsize=16)
    plt.tight_layout()
    plt.show()

# Añade columna time_bucket con etiquetas cada 15 min.
def add_time_bucket_column_spark(
    df,
    frame_col="start_frame",
    kickoff_col="kickoff",
    goal_col="goal",
    frames_per_second=25,
    new_col="time_bucket",
):
    """
    Añade columna time_bucket con etiquetas cada 15 min.
    - Kickoff válido = kickoff==1 y evento anterior goal != 1.
    - Primer kickoff válido = min 0 (1T), segundo = inicio 2T.
    - x = (start_frame - kickoff_frame) / 25 / 60 (minutos en el tiempo actual).
    """
    w_order = Window.orderBy(frame_col)

    df = df.withColumn("_prev_goal", F.lag(F.col(goal_col), 1).over(w_order))
    df = df.withColumn(
        "_valid_kickoff",
        (F.col(kickoff_col) == 1)
        & (F.col("_prev_goal").isNull() | (F.col("_prev_goal") != 1)),
    )

    kickoff_rows = (
        df.filter(F.col("_valid_kickoff"))
        .orderBy(frame_col)
        .limit(2)
        .select(frame_col)
        .collect()
    )
    k0 = int(kickoff_rows[0][0]) if kickoff_rows else 0
    k1 = int(kickoff_rows[1][0]) if len(kickoff_rows) > 1 else None

    fps = float(frames_per_second)
    k0_lit = F.lit(k0)

    df = df.drop("_prev_goal", "_valid_kickoff")

    if k1 is None:
        df = df.withColumn("_half", F.lit(1))
        df = df.withColumn("_x", (F.col(frame_col) - k0_lit) / fps / 60)
    else:
        k1_lit = F.lit(k1)
        df = df.withColumn(
            "_half",
            F.when(F.col(frame_col) >= k1_lit, 2).otherwise(1),
        )
        df = df.withColumn(
            "_x",
            F.when(
                F.col("_half") == 2,
                (F.col(frame_col) - k1_lit) / fps / 60,
            ).otherwise((F.col(frame_col) - k0_lit) / fps / 60),
        )

    df = df.withColumn(
        new_col,
        F.when(F.col("_half") == 1, F.when(F.col("_x") <= 15, F.lit("15 mins"))
            .when(F.col("_x") <= 30, F.lit("30 mins"))
            .when(F.col("_x") <= 45, F.lit("45 mins"))
            .otherwise(F.lit("Tiempo añadido 1T")))
        .otherwise(
            F.when(F.col("_x") <= 15, F.lit("60 mins"))
            .when(F.col("_x") <= 30, F.lit("75 mins"))
            .when(F.col("_x") <= 45, F.lit("90 mins"))
            .otherwise(F.lit("Tiempo añadido 2T")),
        ),
    )
    return df.drop("_half", "_x")
    
def plot_pass_confusion_matrix(df, team_id, fixture_uuid, figsize=(10, 8), annot=True, cmap="Blues"):
    """
    Visualiza una matriz de confusión de pases: filas = pasador, columnas = receptor, valor = nº de pases.
    Mismos filtros que plot_pass_matrix (team_id, fixture_uuid, pass==1).
    Etiquetas: player_name y end_player_name si existen en df; si no, player_id y end_player_id.
    """
    df_f = df[(df["team_id"] == team_id) & (df["fixture_uuid"] == fixture_uuid)]
    df_pas = df_f[df_f["pass"] == 1]

    if df_pas.empty:
        raise ValueError("No hay pases para ese team_id y fixture_uuid.")

    if "player_name" in df_pas.columns and "end_player_name" in df_pas.columns:
        row_col, col_col = "player_name", "end_player_name"
    else:
        row_col, col_col = "player_id", "end_player_id"

    mat = (
        df_pas.groupby([row_col, col_col])
        .size()
        .unstack(fill_value=0)
    )
    # Asegurar que filas y columnas coinciden (todos los jugadores)
    all_players = sorted(set(mat.index) | set(mat.columns))
    mat = mat.reindex(index=all_players, columns=all_players, fill_value=0).astype(int)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        mat,
        annot=annot,
        fmt="d",
        cmap=cmap,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Nº de pases"},
    )
    ax.set_xlabel("Receptor")
    ax.set_ylabel("Pasador")
    ax.set_title("Matriz de pases (pasador → receptor)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()