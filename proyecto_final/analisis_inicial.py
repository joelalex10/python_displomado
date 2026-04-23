"""
PROYECTO FINAL - PARTE I: ANÁLISIS INICIAL DE DATOS
Modelos de Clasificación en Machine Learning
==============================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# PUNTO 1 · Leer los datos
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("PUNTO 1 · Carga del dataset")
print("=" * 70)

df = pd.read_csv("/mnt/user-data/uploads/Datos_pr.csv", encoding="latin1")
print(f"✔ Archivo cargado exitosamente.")
print(f"   Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas\n")


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO 2 · Preguntas de la encuesta (columnas originales en inglés)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("PUNTO 2 · Preguntas de la encuesta (inglés original)")
print("=" * 70)

preguntas_originales = [c for c in df.columns if c not in ("Response Id", "Personality")]
for i, q in enumerate(preguntas_originales, 1):
    print(f"  Q{i:02d}: {q}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO 3 · Traducir nombres de columnas al español
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("PUNTO 3 · Traducción de columnas al español")
print("=" * 70)

traduccion = {
    "Response Id": "ID_Respuesta",
    "You regularly make new friends.":
        "Haces nuevos amigos con regularidad.",
    "You spend a lot of your free time exploring various random topics that pique your interest":
        "Dedicas mucho tiempo libre a explorar temas variados que despiertan tu curiosidad.",
    "Seeing other people cry can easily make you feel like you want to cry too":
        "Ver a otras personas llorar fácilmente te hace querer llorar también.",
    "You often make a backup plan for a backup plan.":
        "A menudo haces un plan de respaldo para tu plan de respaldo.",
    "You usually stay calm, even under a lot of pressure":
        "Generalmente te mantienes calmado/a incluso bajo mucha presión.",
    "At social events, you rarely try to introduce yourself to new people and mostly talk to the ones you already know":
        "En eventos sociales, rara vez intentas presentarte con personas nuevas y prefieres hablar con quienes ya conoces.",
    "You prefer to completely finish one project before starting another.":
        "Prefieres terminar completamente un proyecto antes de comenzar otro.",
    "You are very sentimental.":
        "Eres muy sentimental.",
    "You like to use organizing tools like schedules and lists.":
        "Te gusta usar herramientas de organización como agendas y listas.",
    "Even a small mistake can cause you to doubt your overall abilities and knowledge.":
        "Incluso un pequeño error puede hacerte dudar de tus habilidades y conocimientos.",
    "You feel comfortable just walking up to someone you find interesting and striking up a conversation.":
        "Te sientes cómodo/a acercándote a alguien que te parece interesante para entablar conversación.",
    "You are not too interested in discussing various interpretations and analyses of creative works.":
        "No te interesa mucho discutir distintas interpretaciones y análisis de obras creativas.",
    "You are more inclined to follow your head than your heart.":
        "Te inclinas más a seguir tu cabeza que tu corazón.",
    "You usually prefer just doing what you feel like at any given moment instead of planning a particular daily routine.":
        "Generalmente prefieres hacer lo que se te antoja en el momento en lugar de planificar una rutina diaria.",
    "You rarely worry about whether you make a good impression on people you meet.":
        "Rara vez te preocupa si causas buena impresión en las personas que conoces.",
    "You enjoy participating in group activities.":
        "Disfrutas participar en actividades grupales.",
    "You like books and movies that make you come up with your own interpretation of the ending.":
        "Te gustan los libros y películas que te invitan a crear tu propia interpretación del final.",
    "Your happiness comes more from helping others accomplish things than your own accomplishments.":
        "Tu felicidad proviene más de ayudar a otros a lograr cosas que de tus propios logros.",
    "You are interested in so many things that you find it difficult to choose what to try next.":
        "Te interesan tantas cosas que te resulta difícil elegir qué intentar a continuación.",
    "You are prone to worrying that things will take a turn for the worse.":
        "Tiendes a preocuparte de que las cosas empeoren.",
    "You avoid leadership roles in group settings.":
        "Evitas los roles de liderazgo en entornos grupales.",
    "You are definitely not an artistic type of person.":
        "Definitivamente no eres una persona artística.",
    "You think the world would be a better place if people relied more on rationality and less on their feelings.":
        "Crees que el mundo sería mejor si las personas confiaran más en la razón y menos en sus emociones.",
    "You prefer to do your chores before allowing yourself to relax.":
        "Prefieres hacer tus tareas antes de permitirte relajar.",
    "You enjoy watching people argue.":
        "Disfrutas ver a las personas discutir.",
    "You tend to avoid drawing attention to yourself.":
        "Tiendes a evitar llamar la atención sobre ti mismo/a.",
    "Your mood can change very quickly.":
        "Tu estado de ánimo puede cambiar muy rápidamente.",
    "You lose patience with people who are not as efficient as you.":
        "Pierdes la paciencia con personas que no son tan eficientes como tú.",
    "You often end up doing things at the last possible moment.":
        "A menudo terminas haciendo las cosas en el último momento posible.",
    "You have always been fascinated by the question of what, if anything, happens after death.":
        "Siempre te ha fascinado la pregunta de qué sucede, si es que algo sucede, después de la muerte.",
    "You usually prefer to be around others rather than on your own.":
        "Generalmente prefieres estar rodeado/a de otros antes que estar solo/a.",
    "You become bored or lose interest when the discussion gets highly theoretical.":
        "Te aburres o pierdes el interés cuando la discusión se vuelve muy teórica.",
    "You find it easy to empathize with a person whose experiences are very different from yours.":
        "Te resulta fácil empatizar con una persona cuyas experiencias son muy distintas a las tuyas.",
    "You usually postpone finalizing decisions for as long as possible.":
        "Generalmente pospones la toma de decisiones el mayor tiempo posible.",
    "You rarely second-guess the choices that you have made.":
        "Rara vez cuestionas las decisiones que has tomado.",
    "After a long and exhausting week, a lively social event is just what you need.":
        "Después de una semana larga y agotadora, un evento social animado es justo lo que necesitas.",
    "You enjoy going to art museums.":
        "Disfrutas visitar museos de arte.",
    "You often have a hard time understanding other people\x92s feelings.":
        "A menudo te cuesta entender los sentimientos de otras personas.",
    "You like to have a to-do list for each day.":
        "Te gusta tener una lista de tareas para cada día.",
    "You rarely feel insecure.":
        "Rara vez te sientes inseguro/a.",
    "You avoid making phone calls.":
        "Evitas hacer llamadas telefónicas.",
    "You often spend a lot of time trying to understand views that are very different from your own.":
        "A menudo dedicas mucho tiempo a comprender puntos de vista muy diferentes a los tuyos.",
    "In your social circle, you are often the one who contacts your friends and initiates activities.":
        "En tu círculo social, a menudo eres quien contacta a los amigos e inicia actividades.",
    "If your plans are interrupted, your top priority is to get back on track as soon as possible.":
        "Si tus planes se interrumpen, tu principal prioridad es retomar el rumbo lo antes posible.",
    "You are still bothered by mistakes that you made a long time ago.":
        "Aún te molestan los errores que cometiste hace mucho tiempo.",
    "You rarely contemplate the reasons for human existence or the meaning of life.":
        "Rara vez contemplas las razones de la existencia humana o el significado de la vida.",
    "Your emotions control you more than you control them.":
        "Tus emociones te controlan más de lo que tú las controlas.",
    "You take great care not to make people look bad, even when it is completely their fault.":
        "Tienes mucho cuidado de no hacer quedar mal a las personas, incluso cuando es completamente su culpa.",
    "Your personal work style is closer to spontaneous bursts of energy than organized and consistent efforts.":
        "Tu estilo de trabajo personal se acerca más a ráfagas espontáneas de energía que a esfuerzos organizados y constantes.",
    "When someone thinks highly of you, you wonder how long it will take them to feel disappointed in you.":
        "Cuando alguien piensa bien de ti, te preguntas cuánto tiempo tardará en decepcionarse.",
    "You would love a job that requires you to work alone most of the time.":
        "Te encantaría un trabajo que requiera trabajar solo/a la mayor parte del tiempo.",
    "You believe that pondering abstract philosophical questions is a waste of time.":
        "Crees que reflexionar sobre preguntas filosóficas abstractas es una pérdida de tiempo.",
    "You feel more drawn to places with busy, bustling atmospheres than quiet, intimate places.":
        "Te atraen más los lugares con ambientes animados y concurridos que los lugares tranquilos e íntimos.",
    "You know at first glance how someone is feeling.":
        "Sabes a primera vista cómo se siente alguien.",
    "You often feel overwhelmed.":
        "A menudo te sientes abrumado/a.",
    "You complete things methodically without skipping over any steps.":
        "Completas las cosas de forma metódica sin saltarte ningún paso.",
    "You are very intrigued by things labeled as controversial.":
        "Te intrigan mucho las cosas catalogadas como controvertidas.",
    "You would pass along a good opportunity if you thought someone else needed it more.":
        "Cederías una buena oportunidad si creyeras que alguien más la necesita.",
    "You struggle with deadlines.":
        "Te cuesta cumplir con los plazos.",
    "You feel confident that things will work out for you.":
        "Te sientes seguro/a de que las cosas saldrán bien para ti.",
    "Personality": "Personalidad",
}

df_trad = df.rename(columns=traduccion)
print("✔ Columnas renombradas al español.\n")
print("Primeras columnas del dataframe traducido:")
for col in df_trad.columns[:5]:
    print(f"   • {col}")
print("   ...")
print(f"   • {df_trad.columns[-1]}\n")


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO 4 · Guardar datos traducidos
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("PUNTO 4 · Guardar datos traducidos en datos_trad.csv")
print("=" * 70)

df_trad.to_csv("/mnt/user-data/outputs/datos_trad.csv", index=False, encoding="utf-8-sig")
print("✔ Archivo 'datos_trad.csv' guardado exitosamente.\n")


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO 5 · Número de registros
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("PUNTO 5 · Número de registros")
print("=" * 70)

n_registros = len(df_trad)
n_columnas  = len(df_trad.columns)
n_preguntas = n_columnas - 2   # sin ID y sin Personalidad

print(f"   Total de registros (encuestados): {n_registros:,}")
print(f"   Total de columnas                : {n_columnas}")
print(f"   Preguntas de encuesta            : {n_preguntas}")
print(f"   Columnas de identificación       : ID_Respuesta, Personalidad\n")


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO 6 · Conteo por tipo de personalidad + gráfica
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("PUNTO 6 · Conteo de personas por tipo de personalidad")
print("=" * 70)

conteo = df_trad["Personalidad"].value_counts().sort_values(ascending=False)
print(conteo.to_string())
print()

# Paleta MBTI: 4 dimensiones → colores por grupo temperamental
colores_mbti = {
    "ENTJ": "#C0392B", "INTJ": "#E74C3C", "ENTP": "#E67E22", "INTP": "#F39C12",
    "ENFJ": "#27AE60", "INFJ": "#2ECC71", "ENFP": "#16A085", "INFP": "#1ABC9C",
    "ESTJ": "#2980B9", "ISTJ": "#3498DB", "ESTP": "#8E44AD", "ISTP": "#9B59B6",
    "ESFJ": "#D35400", "ISFJ": "#E74C3C", "ESFP": "#F1C40F", "ISFP": "#F39C12",
}
bar_colors = [colores_mbti.get(p, "#7F8C8D") for p in conteo.index]

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(conteo.index, conteo.values, color=bar_colors, edgecolor="white",
              linewidth=0.8, width=0.7)

# Etiquetas sobre cada barra
for bar, val in zip(bars, conteo.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
            f"{val:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_title("Distribución de Tipos de Personalidad MBTI\n(Encuesta — 60,000 registros)",
             fontsize=15, fontweight="bold", pad=15)
ax.set_xlabel("Tipo de Personalidad", fontsize=12)
ax.set_ylabel("Número de Personas", fontsize=12)
ax.set_ylim(0, conteo.max() * 1.12)
ax.tick_params(axis="x", labelsize=11)
ax.tick_params(axis="y", labelsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/grafica_personalidades.png", dpi=150, bbox_inches="tight")
plt.close()
print("✔ Gráfica guardada como 'grafica_personalidades.png'\n")


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO 7 · Descripción de cada grupo de personalidad
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("PUNTO 7 · Grupos de personalidad identificados (MBTI)")
print("=" * 70)

descripcion_mbti = {
    "INTJ": ("El Arquitecto",
             "Estratega independiente e imaginativo. Planifica con visión a largo plazo, "
             "confía en la lógica y es perfeccionista. Prefiere trabajar solo."),
    "INTP": ("El Lógico",
             "Analítico e innovador, busca patrones y teorías. Disfruta resolver problemas "
             "complejos. Introvertido e intelectualmente curioso."),
    "ENTJ": ("El Comandante",
             "Líder nato, asertivo y orientado a metas. Toma decisiones rápidas y le gusta "
             "dirigir proyectos y equipos con eficiencia."),
    "ENTP": ("El Emprendedor (Innovador)",
             "Creativo y extrovertido, le encanta debatir ideas y encontrar soluciones "
             "originales. Impulsivo y entusiasta ante los desafíos."),
    "INFJ": ("El Abogado (Defensor)",
             "Idealista, empático y reservado. Tiene una visión clara de cómo mejorar el "
             "mundo y lucha por sus valores con determinación."),
    "INFP": ("El Mediador",
             "Soñador, empático y fiel a sus principios. Busca el sentido profundo de las "
             "cosas y valora la autenticidad por encima de todo."),
    "ENFJ": ("El Protagonista",
             "Carismático y orientado a las personas. Inspira y motiva a los demás, con "
             "gran habilidad para entender emociones ajenas."),
    "ENFP": ("El Activista (Campaigner)",
             "Entusiasta, creativo y sociable. Ve el potencial en cada persona y situación, "
             "y disfruta conectar ideas y emociones con los demás."),
    "ISTJ": ("El Inspector (Logístico)",
             "Responsable, ordenado y confiable. Sigue las normas establecidas, es muy "
             "detallista y cumple siempre con sus compromisos."),
    "ISFJ": ("El Defensor (Protector)",
             "Cálido, paciente y servicial. Se preocupa profundamente por el bienestar "
             "ajeno y es fiel a sus tradiciones y responsabilidades."),
    "ESTJ": ("El Ejecutivo",
             "Práctico, decidido y organizador. Valora el orden y la eficiencia; es "
             "excelente administrando sistemas y personas."),
    "ESFJ": ("El Cónsul",
             "Sociable y solícito, vive para ayudar a los demás. Muy sensible a las "
             "necesidades de su entorno y busca armonía en los grupos."),
    "ISTP": ("El Virtuoso",
             "Observador, práctico y tranquilo. Aprende haciendo; le fascina entender "
             "cómo funcionan las cosas y resolver problemas mecánicos."),
    "ISFP": ("El Aventurero",
             "Artístico, sensible y reservado. Vive el momento presente con intensidad y "
             "expresa su creatividad de formas únicas y personales."),
    "ESTP": ("El Emprendedor (Empresario)",
             "Energético, pragmático y audaz. Actúa antes de pensar, disfruta el riesgo "
             "y tiene una gran habilidad para leer a las personas."),
    "ESFP": ("El Animador",
             "Espontáneo, energético y divertido. Ama ser el centro de atención, disfruta "
             "la vida al máximo y es muy generoso con los demás."),
}

for tipo in conteo.index:
    nombre, desc = descripcion_mbti.get(tipo, ("Desconocido", "Sin descripción."))
    print(f"  [{tipo}] {nombre}")
    print(f"         {desc}")
    print(f"         → Registros en el dataset: {conteo[tipo]:,}")
    print()

print("=" * 70)
print("✔ ANÁLISIS INICIAL COMPLETADO EXITOSAMENTE")
print("=" * 70)
