import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.impute import KNNImputer
import nltk
import re
from nltk.stem import WordNetLemmatizer
from scipy.stats import chi2_contingency, f_oneway
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import scipy.stats as stats

if "page" not in st.session_state:
    st.session_state.page = "Home"

st.sidebar.markdown('''Click here to switch between Behavioural and Sales Analysis
''')
if st.sidebar.button("🔄 Behavioural  <-> Sales"):
    st.session_state.page = "Dashboard" if st.session_state.page == "Home" else "Home"


if st.session_state.page == "Home":

    @st.cache_data
    def load_data():
        return pd.read_csv("GamingStudy_data.csv", encoding='latin-1')

    nltk.download('wordnet')
    nltk.download('omw-1.4')

    categories = {
        'Distraction': ['distract', 'escape', 'forget', 'stress', 'anxiety', 'avoid',
                        'reality', 'pain', 'trouble', 'problem', 'depression', 'nervous'],
        'Habit/Time Pass': ['habit', 'time', 'pass', 'bored', 'routine', 'kill time',
                            'occupied', 'waste', 'fill', 'nothing', 'procrastinate'],
        'Social': ['friend', 'team', 'coop', 'multiplayer', 'social', 'together',
                  'community', 'connect', 'bond', 'relationship', 'with others', 'family'],
        'Compete/Win': ['win', 'compete', 'victory', 'rank', 'ladder', 'gm', 'climb',
                        'top', 'leaderboard', 'dominate', 'triumph', 'beat', 'champion'],
        'Improve/Skill': ['improve', 'learn', 'skill', 'progress', 'master', 'practice',
                          'better', 'develop', 'growth', 'hone', 'enhance', 'advance'],
        'Fun/Relax': ['fun', 'relax', 'enjoy', 'chill', 'unwind', 'distress', 'joy',
                      'pleasure', 'entertain', 'distraction', 'happiness']
    }
    category_priority = ['Distraction', 'Habit/Time Pass', 'Social',
                        'Compete/Win', 'Improve/Skill', 'Fun/Relax']

    def categorize_whyplay(text):
        text = re.sub(r'[^\w\s]', '', str(text).lower())
        tokens = text.split()
        verb_tokens = [WordNetLemmatizer().lemmatize(word, pos='v') for word in tokens]
        combined_tokens = set(tokens) | set(verb_tokens)

        matched = []
        for category in category_priority:
            keywords = categories[category]
            for keyword in keywords:
                if any(keyword in token for token in combined_tokens):
                    matched.append(category)
                    break 

        if 'all' in combined_tokens or 'every' in combined_tokens:
            return category_priority
        return matched if matched else ['Other']

    df = load_data()
    df['whyplay_cats'] = df['whyplay'].apply(categorize_whyplay)
    df_exploded = df.explode('whyplay_cats')

    st.title("Data Story on Gaming 🎮")
    
    tab0, tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Introduction",
        "🌟 Reason vs Gender",
        "🌍 Addiction and Satisfaction",
        "📊 Math behind Games",
        "🎮 Game Clusters"
    ])

    with tab0:
        st.subheader("A DATA 605 Project by Ojas and Lalith")
        st.subheader("Why Gaming ??!!")
        st.markdown("""
        Did you know that the biggest industry in terms of revenue isn’t movies or music? Surprisingly,
        it’s the video gaming industry, which generated a staggering 💲187 billion in 2023. In comparison,
        the movie industry brought in 💲133 billion, while the music industry trailed far behind at 💲28 billion.
        That means video games earn more than both industries combined.With such immense growth and potential,
        it's clear that the video game industry is a gold mine worth exploring.
        """)

        st.markdown("""

        1. Gaming Industry
    -	2022: $184 billion (Newzoo)
    Driven by mobile, console, and PC gaming, with strong growth in emerging markets.
    -	2023: $187 billion (estimated)
      Moderate growth due to post-pandemic normalization and mobile dominance.
    -	2024: $200 billion (projected)
      Expected growth from cloud gaming, esports, and next-gen hardware.
    ________________________________________
    2. Music Industry
    -	2022: $26.2 billion (IFPI)
      Streaming accounted for 67% of recorded music revenue.
    -	2023: $28–29 billion (estimated)
      Continued growth in streaming and live events post-pandemic.
    -	2024: $31–33 billion (projected)
      Expansion in emerging markets and vinyl/cassette nostalgia trends.
    ________________________________________
    3. Movie Industry
    -	2022:	Theatrical: \$26 billion (Global Box Office)
      Total (including streaming): $90 billion (MPA estimates).
    -	2023:	Theatrical: \$33 billion (Box Office Mojo)
      Total: $100 billion (streaming platforms like Netflix and Disney+).
    -	2024: $105–110 billion (projected)
      Hybrid releases (theatrical + streaming) and international markets driving growth.
    ________________________________________
    4. Sports Industry
    -	2022: $487 billion (Statista)
      Includes media rights, sponsorships, merchandise, and live events.
    -	2023: $500–520 billion (estimated)
      Recovery in live attendance and rising media deals (e.g., NFL, Premier League).
    -	2024: $550–600 billion (projected)
      Growth in digital streaming rights and global events (e.g., Olympics, FIFA World Cup)

        """)

 
    with tab1:
        st.header("Behavioral Analysis")

        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            gender_options = ["All"] + df['Gender'].dropna().unique().tolist()
            selected_gender = st.radio(
                "Gender",
                gender_options,
                index=0,
                key="gender_filter"
            )

        filtered_exploded = df_exploded.copy()

        if selected_gender != "All":
            filtered_exploded = filtered_exploded[filtered_exploded['Gender'] == selected_gender]

        fig = px.bar(filtered_exploded['whyplay_cats'].value_counts().reset_index(),
                    x='count', y='whyplay_cats', orientation='h',
                    title=f"Gaming Motivations Breakdown",
                    labels={'whyplay_cats': 'Motivation Category', 'count': 'Player Count'},
                    color='whyplay_cats',
                    color_discrete_sequence=px.colors.qualitative.Pastel)

        filter_text = []
        if selected_gender != "All": filter_text.append(f"Gender: {selected_gender}")


        if filter_text:
            st.caption(f"Active filters: {', '.join(filter_text)}")

        fig.update_layout(
            height=500,
            width=800,
            yaxis_title="Motivation Category",
            xaxis_title="Number of Players",
            legend_title="Motivation",
            showlegend=False  
        )

        st.plotly_chart(fig)
        st.markdown("""
        - Males tend to report more competitive motivations ("Compete/Win") and "Improve/Skill".
        - Female players emphasize "Fun/Relax" aspects than their counterparts
        """)
        st.header("Statistical Analysis")
        contingency_table = pd.crosstab(df_exploded['Gender'], df_exploded['whyplay_cats'])
        chi2, p, _, _ = chi2_contingency(contingency_table)

        st.subheader("Chi-Square Test Results")
        st.markdown(f"""
        - **Chi-Square Statistic**: `{chi2:.2f}`
        - **P-value**: `{p:.20f}`
        - **Significance**: {'✅ Significant' if p < 0.05 else '❌ Not Significant'}
        """)
        st.subheader("Key Insights")


        st.subheader("Interpretation:")
        st.markdown("""
        There is a statistically significant association between gender and gaming motivations. This means:
        - Males and females have different motivational patterns when gaming.
        - The observed differences (e.g., males emphasizing "Compete/Win," females prioritizing "Fun/Relax") are unlikely due to random chance (p < 0.05).
        """)


    # ------------------------------------------------------------------------------

    df_original = df.copy()

    features_hours = ["streams", "SPIN_T", "SPIN13", "SPIN16", "SPIN12",
                      "Narcissism", "SPIN8", "SPIN10", "SPIN3", "SPIN14"]
    df_filtered = df.dropna(subset=["Hours"] + features_hours)
    X_train = df_filtered[features_hours]
    y_train = df_filtered["Hours"]
    imputer = KNNImputer(n_neighbors=2)
    df[features_hours] = imputer.fit_transform(df[features_hours])
    df = df[~df["Hours"].isin([420, 8000])]

    features_narc = ["GAD6", "GAD_T", "GAD5"]
    df_filtered = df.dropna(subset=["Narcissism"] + features_narc)
    X_train = df_filtered[features_narc]
    y_train = df_filtered["Narcissism"]
    imputer = KNNImputer(n_neighbors=2)
    df[features_narc] = imputer.fit_transform(df[features_narc])

    features_streams = ["Hours", "SPIN_T"]
    df_filtered = df.dropna(subset=["streams"] + features_streams)
    X_train = df_filtered[features_streams]
    y_train = df_filtered["streams"]
    imputer = KNNImputer(n_neighbors=2)
    df[features_streams] = imputer.fit_transform(df[features_streams])

    s_drop = ['Unnamed: 0', 'Zeitstempel', 'Birthplace_ISO3', 'Residence_ISO3', 'highestleague',
              'GAD1', 'GAD2', 'GAD3', 'GAD4', 'GAD5', 'GAD6', 'GAD7',
              'SWL1', 'SWL2', 'SWL3', 'SWL4', 'SWL5',
              'SPIN1', 'SPIN2', 'SPIN3', 'SPIN4', 'SPIN5', 'SPIN6', 'SPIN7', 'SPIN8',
              'SPIN9', 'SPIN10', 'SPIN11', 'SPIN12', 'SPIN13', 'SPIN14', 'SPIN15',
              'SPIN16', 'SPIN17', 'SPIN_T', 'accept']
    df = df.drop(columns=s_drop, errors='ignore')


    with tab2:

        st.header("Gaming Hours vs Anxiety & Satisfaction with Life")

        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                gender_filter = st.selectbox("Select Gender", ["All"] + df["Gender"].dropna().unique().tolist(), key="gender_filter_tab2")
                work_type = st.selectbox("Select Work Type", ["All"] + df["Work"].dropna().unique().tolist(), key="work_type_tab2")
            with col2:
                degree = st.selectbox("Select Degree", ["All"] + df["Degree"].dropna().unique().tolist(), key="degree_tab2")
                residence = st.selectbox("Select Residence", ["All"] + df["Residence"].dropna().unique().tolist(), key="residence_tab2")
                game_type = st.selectbox("Select Game Type", ["All"] + df["Game"].dropna().unique().tolist(), key="game_type_tab2")

        filtered_df = df.copy()
        if gender_filter != "All":
            filtered_df = filtered_df[filtered_df["Gender"] == gender_filter]
        if work_type != "All":
            filtered_df = filtered_df[filtered_df["Work"] == work_type]
        if degree != "All":
            filtered_df = filtered_df[filtered_df["Degree"] == degree]
        if residence != "All":
            filtered_df = filtered_df[filtered_df["Residence"] == residence]
        if game_type != "All":
            filtered_df = filtered_df[filtered_df["Game"] == game_type]

        if "Hours" in filtered_df.columns and "GAD_T" in filtered_df.columns:
            fig1 = px.scatter(filtered_df, x="Hours", y="GAD_T", trendline="ols",
                              title="Gaming Hours vs Anxiety (GAD_T)",
                              color_discrete_sequence=['#FF6347'])
            fig1.update_traces(line=dict(color='green'))
            st.plotly_chart(fig1)

        st.markdown('''More gaming hours tend to correlate with slightly higher anxiety, but the effect is weak.​
        ''')

        if "Hours" in filtered_df.columns and "SWL_T" in filtered_df.columns:
            fig2 = px.scatter(filtered_df, x="Hours", y="SWL_T", trendline="ols",
                              title="Gaming Hours vs Satisfaction with Life",
                              color_discrete_sequence=['#1E90FF'])
            fig2.update_traces(line=dict(color='red'))
            st.plotly_chart(fig2)

        st.markdown('''Higher gaming hours are associated with lower life satisfaction, but the effect is not uniform for all gamers.
        ''')




    with tab3:

        numeric_df_original = df_original.drop(columns=s_drop, errors='ignore').select_dtypes(include=['number'])
        correlation_matrix_original = numeric_df_original.corr()
        st.subheader("📊 Correlation Table ")
        st.dataframe(correlation_matrix_original.style.format("{:.4f}").background_gradient(cmap='coolwarm'))
        
        st.markdown('''Gaming Hours & Streams (0.9761, Strong Positive Correlation)​ This suggests that i
        ndividuals who spend more hour's gaming also tend to watch streams or other gameplay more frequently.
        ''')

        st.markdown('''Anxiety & Life Satisfaction (-0.4033, Moderate Negative Correlation)​
        This is one of the strongest correlations in the table, showing that higher anxiety is associated with lower life satisfaction.
        ''')    

        fig = px.box(df, x='Work', y='SWL_T',
                    title="Life Satisfaction (SWL_T) by Employment Status",
                    labels={"Work": "Employment Status", "SWL_T": "Life Satisfaction (SWL_T)"})
        st.plotly_chart(fig)
        st.markdown("""
        **Insight:** Employed individuals reported higher life satisfaction (Mean = 20.7) compared to unemployed individuals (Mean = 14.7).
        The ANOVA indicates that these differences are statistically significant (F = 274.84, p = 0.00).
        """)

        groups = [group['SWL_T'].dropna() for name, group in df.groupby('Work')]

        F, p = stats.f_oneway(*groups)

        st.write("### ANOVA Results")
        st.write(f"F-statistic = {F:.2f}, p-value = {p:.3f}")

        st.subheader("Interpretation:")
        st.markdown("""
        Life satisfaction (SWL_T) varies significantly across employment statuses:

        - Large F-value (274.84) indicates strong group differences.

        - Employed individuals (Mean SWL_T = 20.7) report higher life satisfaction than unemployed (14.7).

        - Practical implication: Unemployment may exacerbate mental health challenges in gamers, while employment correlates with better well-being.
        """)

        mean_values = df.groupby('Work')['SWL_T'].mean().round(1)
        st.write("### Mean Life Satisfaction (SWL_T) by Employment Status")
        st.write(mean_values)



    # -------------------------------------------------

    with tab4:
        st.title("🔍 Gamers Clustering Based on Habits and Mental Health")

        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA

        cluster_vars = ['Hours', 'GAD_T', 'SWL_T']

        clustering_data = df[cluster_vars].dropna()

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clustering_data)

        kmeans = KMeans(n_clusters=2, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        clustering_data['Cluster'] = clusters

        pca = PCA(n_components=2, random_state=42)
        pca_components = pca.fit_transform(scaled_data)
        clustering_data['PC1'] = pca_components[:, 0]
        clustering_data['PC2'] = pca_components[:, 1]

        fig_cluster = px.scatter(clustering_data, x='PC1', y='PC2', color='Cluster',
                                title="Gamers Clusters (PCA-Reduced)",
                                labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"},
                                color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig_cluster)

        cluster_profiles = clustering_data.groupby('Cluster')[cluster_vars].mean().round(1)
        st.write("### Cluster Profiles")
        st.dataframe(cluster_profiles)

        st.markdown("""
        Real-World Analogy:
        - Cluster 0 (Balanced Players)

          Gaming Hours: Moderate (18.9 hours)

          Anxiety (GAD_T): Low (2.8)

          Life Satisfaction (SWL_T): High (23.7)

          Profile: Players with healthy gaming habits who maintain good mental health and life satisfaction. Likely play for enjoyment/skill development without excessive time commitment.

        - Cluster 1 (At-Risk Players)

          Gaming Hours: High (26 hours)

          Anxiety (GAD_T): Elevated (9.1)

          Life Satisfaction (SWL_T): Low (13.4)

          Profile: Players showing potential signs of problematic gaming behavior - longer playtimes correlate with higher anxiety and reduced life satisfaction. May be using gaming as an escape mechanism.
        """)

        st.subheader(" Key Psychological Insight:")
        st.markdown("""

        The pattern shows an inverse relationship between gaming hours and mental health metrics:

        ↑ More gaming hours = ↑ Anxiety + ↓ Life satisfaction

        ↓ Moderate gaming = ↓ Anxiety + ↑ Life satisfaction

        This aligns with clinical observations that excessive gaming can be both a symptom and contributor to mental health challenges.

        """)

        #----------------------oj---------------------
else:



    st.markdown("""
    <style>
    @keyframes fadeIn {
      from {opacity: 0;}
      to {opacity: 1;}
    }
    .fade-in {
      animation: fadeIn 1s ease-in;
    }
    </style>
    """, unsafe_allow_html=True)

    @st.cache_data
    def load_data():
        df = pd.read_csv("Video_Games_Sales_as_at_22_Dec_2016.csv")
        df["User_Score"] = pd.to_numeric(df["User_Score"], errors='coerce')
        median_year = int(df["Year_of_Release"].median())
        df["Year_of_Release"].fillna(median_year, inplace=True)


        numeric_columns = ["Critic_Score", "Critic_Count", "User_Score", "User_Count"]
        imputer = KNNImputer(n_neighbors=5)
        df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

        return df

    df = load_data()

    selected_year_range = st.sidebar.slider(
        "Select Year Range",
        min_value=int(df['Year_of_Release'].min()),
        max_value=int(df['Year_of_Release'].max()),
        value=(2000, 2016)
    )

    selected_genres = st.sidebar.multiselect(
        "Select Genres",
        options=df['Genre'].unique(),
        default=['Action', 'Sports', 'Shooter']
    )

    st.title("Data Story on Gaming 🎮")
    st.header(" 💲 Video Game Sales")
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)

    filtered_df = df[
        (df['Year_of_Release'].between(*selected_year_range)) &
        (df['Genre'].isin(selected_genres))
    ]


    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Games Analyzed", len(filtered_df))
    with col2:
        st.metric("Total Global Sales", f"${filtered_df['Global_Sales'].sum():,.1f}B")
    with col3:
        st.metric("Average Critic Score", f"{filtered_df['Critic_Score'].mean():.1f}/100")


    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Sales Analysis",
        "🌟 Score Correlations",
        "🌍 Regional Insights",
        "🎮 Game Clusters",
        "📝 References"
    ])

    with tab1:
        st.header("Sales Performance Analysis")


        num_games = st.slider("Select Number of Top Games", 5, 20, 10)
        top_games = filtered_df.nlargest(num_games, 'Global_Sales')

        fig = px.bar(top_games,
                    x='Global_Sales',
                    y='Name',
                    orientation='h',
                    color='Platform',
                    title=f"Top {num_games} Best-Selling Games",
                    labels={'Global_Sales': 'Global Sales (Millions)'})
        st.plotly_chart(fig, use_container_width=True)




        st.markdown('''Wii Dominates:The Wii platform (blue bars) has the highest-selling games, with Wii 
        Sports being the best-selling title, surpassing 80 million units.
        Other popular Wii games include Wii Sports Resort, Wii Fit, and Wii Fit Plus.
        Grand Theft Auto (GTA) Series is Strong:
        
        GTA V is the highest-selling multi-platform game (PS3 + X360), with a strong presence in light blue (PS3) and pink (X360).
        GTA: San Andreas and GTA: Vice City have high sales, particularly on PS2 (red bars).
        Call of Duty's Popularity:
        
        Call of Duty: Modern Warfare 3 (X360 - pink) and Call of Duty: Black Ops 3 (PS4 - green) are among the top 10.
        Despite high sales, they are lower than the best-selling Wii and GTA titles.
        ''')

        st.markdown('''- Wii had some of the highest-selling exclusive games, likely due to its popularity as a casual gaming console.
        - Multi-platform titles like GTA V and Call of Duty performed well across different consoles.
        - Older generation consoles like PS2 (GTA titles) still made an impact in terms of total sales.
        ''')
        st.subheader("Sales Trends Over Time")
        trend_data = filtered_df.groupby("Year_of_Release")["Global_Sales"].sum().reset_index()
        fig = px.area(trend_data,
                      x='Year_of_Release',
                      y='Global_Sales',
                      markers=True,
                      title="Global Sales Trend Over Time")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('''Growth Phase (2000 - 2008) 
        Sales increased steadily from 2000 to 2008, peaking around 2008.
        The gaming industry saw significant growth, likely due to the rise of popular gaming consoles (PS2, PS3, Xbox 360, Wii) and blockbuster titles.
        
        Peak Period (2008 - 2009)
        Global sales hit the highest point in 2008-2009, surpassing 350 million units.
        This could be attributed to the dominance of Wii, PlayStation, and Xbox franchises, along with popular games like Grand Theft Auto IV, Call of Duty: Modern Warfare, and Wii Sports.
        
        Decline Phase (2010 - 2016)
        After 2009, sales gradually declined, dropping significantly post-2012.
        Possible reasons:
        Market saturation and fewer groundbreaking game releases.
        Transition to digital downloads and online gaming, reducing physical sales.
        The decline of older console sales (PS2, Wii, Xbox 360).
        Steady Drop After 2014
        
        Post-2014, the decline becomes sharper.
        Could be linked to the rise of mobile gaming, digital storefronts (Steam, PlayStation Store, Xbox Live), reducing reliance on physical sales.
        ''')
        st.markdown('''- Golden Era: 2006-2009 was the peak of global game sales.
        - Shift in Industry Trends: The decline post-2010 suggests a move from physical game sales to digital downloads and live services.
        - Modern Trends: If extended to 2020s, digital sales (e.g., on Steam, PSN, and Xbox Live) and subscription services (Game Pass, PlayStation Plus) would likely dominate.
        ''')

    with tab2:
        st.header("Review Score Analysis")


        score_type = st.radio("Select Score Type", ['Critic_Score', 'User_Score'])

        fig = px.scatter(filtered_df,
                        x=score_type,
                        y='Global_Sales',
                        color='Genre',
                        size='Critic_Count',
                        hover_name='Name',
                        title=f"{score_type.replace('_', ' ')} vs Global Sales")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('''Higher Critic Scores Don’t Always Guarantee Higher Sales

Many games with critic scores above 80 still have low global sales.
However, some games with high sales do have strong critic scores (above 80), showing a slight correlation.
Sports Games (Dark Blue)

A few outliers with extremely high global sales (~80 million) despite varying critic scores.
These could be popular franchises like FIFA, Madden NFL, or Wii Sports, which have strong brand recognition regardless of reviews.
Action Games (Light Blue)

Several high-selling games appear in this category, especially for scores above 70.
Some action games with very high critic scores (near 100) still show moderate sales, suggesting good reviews don’t always translate to massive commercial success.
Shooter Games (Red)

Most populated genre in the dataset.
Several games with moderate to high scores (70-90) have significant sales (5-20 million).
Likely includes franchises like Call of Duty and Battlefield, which sell well even with mixed reviews.
        ''')

        st.markdown('''
        - Sports games can have huge sales despite varying critic scores (likely due to brand loyalty).
- Shooter and Action games show a stronger correlation between high critic scores and higher sales.
- Games with a score below ~60 generally have low sales, reinforcing that poor reviews can negatively impact commercial success.
        ''')

        st.subheader("Regional Sales Correlations")
        corr_matrix = filtered_df[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].corr()
        fig = px.imshow(corr_matrix,
                        text_auto=True,
                        color_continuous_scale='Blues',
                        title="Regional Sales Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("Regional Market Analysis")


        region = st.selectbox("Select Region", ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'])
        regional_data = filtered_df.groupby('Genre')[region].sum().reset_index()

        fig = px.pie(regional_data,
                    names='Genre',
                    values=region,
                    title=f"{region.replace('_', ' ')} Distribution by Genre")
        st.plotly_chart(fig, use_container_width=True)


        st.subheader("Top Publishers by Region")
        publishers = filtered_df.groupby('Publisher')[['NA_Sales', 'EU_Sales', 'JP_Sales']].sum()
        fig = px.bar(publishers.nlargest(5, 'NA_Sales'),
                    orientation='h',
                    title="Top Publishers in North America")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.header("Game Clustering Analysis")


        n_clusters = st.slider("Select Number of Clusters", 2, 5, 3)

        cluster_df = filtered_df[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].dropna()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_df)


        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        cluster_df['Cluster'] = kmeans.fit_predict(scaled_data)


        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(scaled_data)
        cluster_df['PC1'] = pca_results[:, 0]
        cluster_df['PC2'] = pca_results[:, 1]

        fig = px.scatter(cluster_df,
                        x='PC1',
                        y='PC2',
                        color='Cluster',
                        hover_data=['NA_Sales', 'EU_Sales'],
                        title="PCA Visualization of Game Clusters")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Cluster Characteristics")
        profile = cluster_df.groupby('Cluster').mean()
        st.dataframe(profile.style.background_gradient(cmap='Blues'), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


    with tab5:

        st.header("""Thank you""")
        st.header("References")
        st.markdown("""
        - The Gaming sales dataset from [Kaggle Dataset](https://www.kaggle.com/datasets/gregorut/videogamesales)

        - The Gaming Psychological dataset from Open Science Framework (https://osf.io/vnbxk/)

        """)




