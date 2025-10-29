import pandas as pd
import numpy as np
import hashlib
import random
import re
from datetime import datetime, timedelta

# 1. Generar dataset de usuarios

nombres = ["Juan", "Camila", "Andrés", "María", "Sofía", "Pedro", "Luis", "Ana", "Valentina", "Carlos"]
apellidos = ["Gómez", "Martínez", "Pérez", "Rodríguez", "López", "Hernández", "García", "Ramírez"]

roles = ["admin", "estudiante", "profesor"]

usuarios = []
for i in range(50):  # Cantidad aceptable de usuarios simulados
    nombre = random.choice(nombres) + " " + random.choice(apellidos)
    correo = nombre.lower().replace(" ", ".") + f"{random.randint(1,99)}@gmail.com"
    contraseña = hashlib.sha256(f"pass{i}".encode()).hexdigest()  # Hash seguro
    rol = random.choice(roles)
    fecha_registro = datetime.now() - timedelta(days=random.randint(1, 1000))
    
    usuarios.append([nombre, correo, contraseña, rol, fecha_registro])

df_users = pd.DataFrame(usuarios, columns=["nombre", "correo", "contraseña", "rol", "fecha_registro"])

# Guardar en CSV
df_users.to_csv("usuarios.csv", index=False)

print("=== Dataset de Usuarios ===")
print(df_users.head())

# 2. Filtrar duplicados y correos inválidos con regex

correo_regex = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"

df_users["correo_valido"] = df_users["correo"].apply(lambda x: bool(re.match(correo_regex, x)))
df_clean = df_users[df_users["correo_valido"]]  # solo válidos
df_clean = df_clean.drop_duplicates(subset=["correo"])

print("\n=== Usuarios limpios (sin duplicados ni correos inválidos) ===")
print(df_clean.head())


# 3. Dataset simulado de hojas de vida

perfiles = ["Backend Developer", "Frontend Developer", "Fullstack", "Data Scientist"]
habilidades = {
    "Backend Developer": ["Python", "Django", "Flask", "SQL"],
    "Frontend Developer": ["React", "JavaScript", "CSS", "HTML"],
    "Fullstack": ["Node.js", "React", "MongoDB", "Express"],
    "Data Scientist": ["Python", "Pandas", "NumPy", "Machine Learning"]
}

hojas_vida = []
for i in range(50):
    nombre = random.choice(nombres) + " " + random.choice(apellidos)
    edad = random.randint(18, 40)
    perfil = random.choice(perfiles)
    skills = random.sample(habilidades[perfil], k=random.randint(2, 4))
    certificados = random.sample(["AWS", "Azure", "Google Cloud", "Scrum", "Docker"], k=random.randint(0, 3))
    proyectos = [f"https://github.com/user/proyecto{i}" for i in range(random.randint(1, 3))]
    resumen = " ".join(["Experiencia en"] + skills + ["con proyectos académicos y laborales."])
    semestre = random.randint(1, 10)

    hojas_vida.append([nombre, edad, perfil, skills, certificados, proyectos, resumen, semestre])

df_cv = pd.DataFrame(hojas_vida, columns=["nombre", "edad", "perfil", "habilidades", "certificados", "proyectos", "resumen", "semestre"])

print("\n=== Dataset de Hojas de Vida ===")
print(df_cv.head())


# Longitud promedio de resumen por semestre
df_cv["len_resumen"] = df_cv["resumen"].apply(len)
promedio_por_semestre = df_cv.groupby("semestre")["len_resumen"].mean()

print("\n=== Longitud promedio del resumen por semestre ===")
print(promedio_por_semestre)

# Detectar duplicados de certificados
df_cv["certificados_str"] = df_cv["certificados"].astype(str)
duplicados_cert = df_cv[df_cv.duplicated("certificados_str", keep=False)]

print("\n=== Hojas de vida con certificados duplicados ===")
print(duplicados_cert[["nombre", "certificados"]])

# Agrupar estudiantes por tecnologías dominantes
tech_count = df_cv.explode("habilidades").groupby("habilidades")["nombre"].count().sort_values(ascending=False)

print("\n=== Agrupación por habilidades dominantes ===")
print(tech_count)

# Filtrar hojas de vida incompletas (sin proyectos o certificados)
incompletas = df_cv[(df_cv["proyectos"].str.len() == 0) | (df_cv["certificados"].str.len() == 0)]

print("\n=== Hojas de vida incompletas ===")
print(incompletas)



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------------
# Generar dataset simulado
# -----------------------------------

np.random.seed(42)

n = 30  # cantidad de estudiantes

nombres = [f"Estudiante_{i}" for i in range(1, n + 1)]
edades = np.random.randint(18, 35, n)
semestres = np.random.choice([1, 3, 5, 7, 9], n)
habilidades = np.random.choice([
    "Python, Django, SQL",
    "JavaScript, React, CSS",
    "Java, Spring Boot, SQL",
    "HTML, CSS, JS",
    "Node.js, Express, MongoDB",
    "React, TypeScript, Redux"
], n)
certificados = np.random.choice([
    "AWS", "Azure", "Google Cloud", "Scrum", "AWS", "SQL"
], n)
proyectos = np.random.choice([
    "E-commerce", "Portfolio", "API REST", "Dashboard", "Portfolio"
], n)
resumenes = np.random.choice([
    "Desarrollador con pasión por crear soluciones eficientes.",
    "Interesado en desarrollo web moderno.",
    "Experiencia en backend con APIs escalables.",
    "Enfocado en frontend con React y UI/UX.",
    "Estudiante con interés en arquitectura de software."
], n)
completitud = np.random.choice([True, True, True, False], n)  # algunos incompletos

df = pd.DataFrame({
    "nombre": nombres,
    "edad": edades,
    "semestre": semestres,
    "habilidades": habilidades,
    "certificados": certificados,
    "proyectos": proyectos,
    "resumen": resumenes,
    "completo": completitud
})

print("=== Dataset simulado ===")
print(df.head(), "\n")

# -----------------------------------
# Clasificar perfiles automáticamente
# -----------------------------------

def clasificar_perfil(hab):
    if "React" in hab or "JS" in hab:
        return "Frontend"
    elif "Python" in hab or "Django" in hab:
        return "Backend"
    elif "Java" in hab or "Spring" in hab:
        return "Backend"
    elif "Node" in hab or "Express" in hab:
        return "Full Stack"
    else:
        return "Otro"

df["perfil"] = df["habilidades"].apply(clasificar_perfil)

# -----------------------------------
# Comparar longitud promedio del resumen por semestre
# -----------------------------------

df["longitud_resumen"] = df["resumen"].apply(len)

promedio_por_semestre = df.groupby("semestre")["longitud_resumen"].mean().reset_index()

plt.figure(figsize=(7,4))
sns.barplot(data=promedio_por_semestre, x="semestre", y="longitud_resumen", palette="mako")
plt.title("Longitud promedio del resumen profesional por semestre")
plt.show()

# -----------------------------------
# Detectar duplicados en certificados o proyectos
# -----------------------------------

duplicados_cert = df[df.duplicated("certificados", keep=False)]
duplicados_proy = df[df.duplicated("proyectos", keep=False)]

print("=== Certificados duplicados ===")
print(duplicados_cert[["nombre", "certificados"]].sort_values("certificados"), "\n")

print("=== Proyectos duplicados ===")
print(duplicados_proy[["nombre", "proyectos"]].sort_values("proyectos"), "\n")

# -----------------------------------
# Agrupar por tecnologías dominantes
# -----------------------------------

conteo_perfil = df["perfil"].value_counts().reset_index()
conteo_perfil.columns = ["perfil", "cantidad"]

plt.figure(figsize=(6,4))
sns.barplot(data=conteo_perfil, x="perfil", y="cantidad", palette="Set2")
plt.title("Distribución de estudiantes por perfil tecnológico")
plt.show()

# -----------------------------------
# Filtrar hojas de vida incompletas
# -----------------------------------

incompletos = df[df["completo"] == False]
print("=== Hojas de vida incompletas ===")
print(incompletos[["nombre", "completo"]], "\n")

# -----------------------------------
# Mostrar resumen general
# -----------------------------------

plt.figure(figsize=(8,5))
sns.countplot(data=df, x="semestre", hue="perfil", palette="coolwarm")
plt.title("Perfiles por semestre académico")
plt.show()


# Porcentaje de hojas de vida completas vs incompletas
completo_counts = df["completo"].value_counts()
plt.figure(figsize=(5,5))
plt.pie(
    completo_counts,
    labels=["Completas", "Incompletas"],
    autopct="%1.1f%%",
    startangle=90,
    colors=["#66b3ff", "#ff9999"]
)
plt.title("Porcentaje de hojas de vida completas vs incompletas")
plt.show()


#  Edad promedio por semestre
edad_semestre = df.groupby("semestre")["edad"].mean().reset_index()
plt.figure(figsize=(7,4))
sns.lineplot(data=edad_semestre, x="semestre", y="edad", marker="o", color="orange")
plt.title("Edad promedio por semestre académico")
plt.xlabel("Semestre")
plt.ylabel("Edad promedio")
plt.show()