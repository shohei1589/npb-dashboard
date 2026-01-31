DROP VIEW IF EXISTS batting_1_view;

CREATE VIEW batting_1_view AS
SELECT
  年度,
  所属,

  選手名,
  選手ID,
  年齢,
  投,
  打,

  試合,
  打席,
  打数,
  得点,
  安打,
  二塁打,
  三塁打,
  本塁打,
  塁打,
  打点,
  三振,
  四球,
  敬遠,
  死球,
  犠打,
  犠飛,
  盗塁,
  盗塁死,
  併殺打,

  /* --- 得点圏打率（2016年以降のみ） --- */
  CASE
    WHEN 年度 >= 2016 THEN 得点圏打率
    ELSE NULL
  END AS 得点圏打率,

  /* --- 打率 --- */
  CASE WHEN 打数 > 0
       THEN 安打 * 1.0 / 打数
  END AS 打率,

  /* --- 出塁率 --- */
  CASE WHEN (打数 + 四球 + 死球 + 犠飛) > 0
       THEN (安打 + 四球 + 死球) * 1.0
            / (打数 + 四球 + 死球 + 犠飛)
  END AS 出塁率,

  /* --- 長打率 --- */
  CASE WHEN 打数 > 0
       THEN (
         (安打 - 二塁打 - 三塁打 - 本塁打)
         + 2*二塁打 + 3*三塁打 + 4*本塁打
       ) * 1.0 / 打数
  END AS 長打率,

  /* --- ISO --- */
  CASE WHEN 打数 > 0
       THEN (二塁打 + 2*三塁打 + 3*本塁打) * 1.0 / 打数
  END AS ISO,

  /* --- BABIP --- */
  CASE WHEN (打数 - 三振 - 本塁打 + 犠飛) > 0
       THEN (安打 - 本塁打) * 1.0
            / (打数 - 三振 - 本塁打 + 犠飛)
  END AS BABIP,

  /* --- K% / BB% --- */
  CASE WHEN 打席 > 0
       THEN 三振 * 100.0 / 打席
  END AS "K%",

  CASE WHEN 打席 > 0
       THEN 四球 * 100.0 / 打席
  END AS "BB%",

  /* --- BB/K --- */
  CASE WHEN 三振 > 0
       THEN 四球 * 1.0 / 三振
  END AS "BB/K",

  /* --- wOBA（失策出塁なし） --- */
  CASE WHEN (打数 + 四球 - 敬遠 + 死球 + 犠飛) > 0
       THEN (
         0.692 * (四球 - 敬遠)
         + 0.73  * 死球
         + 0.865 * (安打 - 二塁打 - 三塁打 - 本塁打)
         + 1.334 * 二塁打
         + 1.725 * 三塁打
         + 2.065 * 本塁打
       ) * 1.0
       / (打数 + 四球 - 敬遠 + 死球 + 犠飛)
  END AS wOBA,

  /* --- OPS --- */
  CASE WHEN (打数 + 四球 + 死球 + 犠飛) > 0 AND 打数 > 0
       THEN
         ((安打 + 四球 + 死球) * 1.0
          / (打数 + 四球 + 死球 + 犠飛))
         +
         (
           ((安打 - 二塁打 - 三塁打 - 本塁打)
            + 2*二塁打 + 3*三塁打 + 4*本塁打)
           * 1.0 / 打数
         )
  END AS OPS,

  /* --- Spd (Speed Score) --- */
  (
    COALESCE(
      ( ((盗塁 + 3.0) / (盗塁 + 盗塁死 + 7.0)) - 0.4 ) * 20.0,
      0
    )
    +
    COALESCE(
      CASE WHEN (安打 - 二塁打 - 三塁打 - 本塁打 + 四球 + 死球) > 0
           THEN sqrt(
             (盗塁 + 盗塁死) * 1.0
             / (安打 - 二塁打 - 三塁打 - 本塁打 + 四球 + 死球)
           ) / 0.07
      END,
      0
    )
    +
    COALESCE(
      CASE WHEN (打数 - 本塁打 - 三振) > 0
           THEN (三塁打 * 1.0 / (打数 - 本塁打 - 三振))
                / 0.02 * 10.0
      END,
      0
    )
    +
    COALESCE(
      CASE WHEN (安打 + 四球 + 死球 - 本塁打) > 0
           THEN (
             ((得点 - 本塁打) * 1.0
              / (安打 + 四球 + 死球 - 本塁打))
             - 0.1
           ) / 0.04
      END,
      0
    )
  ) / 4.0 AS Spd

FROM batting_1_raw;


DROP VIEW IF EXISTS batting_2_view;

CREATE VIEW batting_2_view AS
SELECT
  年度,
  所属,

  選手名,
  選手ID,
  年齢,
  投,
  打,

  試合,
  打席,
  打数,
  得点,
  安打,
  二塁打,
  三塁打,
  本塁打,
  塁打,
  打点,
  三振,
  四球,
  敬遠,
  死球,
  犠打,
  犠飛,
  盗塁,
  盗塁死,
  併殺打,

  -- 2軍には得点圏打率が無いのでNULL固定
  NULL AS 得点圏打率,


  -- 率系
  CASE WHEN 打数 > 0 THEN 安打 * 1.0 / 打数 END AS 打率,

  CASE WHEN (打数 + 四球 + 死球 + 犠飛) > 0
       THEN (安打 + 四球 + 死球) * 1.0 / (打数 + 四球 + 死球 + 犠飛)
  END AS 出塁率,

  CASE WHEN 打数 > 0
       THEN ((安打 - 二塁打 - 三塁打 - 本塁打)
            + 2*二塁打 + 3*三塁打 + 4*本塁打) * 1.0 / 打数
  END AS 長打率,

  CASE WHEN 打数 > 0
       THEN (二塁打 + 2*三塁打 + 3*本塁打) * 1.0 / 打数
  END AS ISO,

  CASE WHEN (打数 - 三振 - 本塁打 + 犠飛) > 0
       THEN (安打 - 本塁打) * 1.0
            / (打数 - 三振 - 本塁打 + 犠飛)
  END AS BABIP,

  CASE WHEN 打席 > 0 THEN 三振 * 100.0 / 打席 END AS "K%",
  CASE WHEN 打席 > 0 THEN 四球 * 100.0 / 打席 END AS "BB%",

  CASE WHEN (打数 + 四球 + 死球 + 犠飛) > 0 AND 打数 > 0
       THEN
         ((安打 + 四球 + 死球) * 1.0 / (打数 + 四球 + 死球 + 犠飛))
         +
         (((安打 - 二塁打 - 三塁打 - 本塁打)
           + 2*二塁打 + 3*三塁打 + 4*本塁打) * 1.0 / 打数)
  END AS OPS,

  -- wOBA（失策出塁は入れない版）
  CASE WHEN (打数 + 四球 - 敬遠 + 死球 + 犠飛) > 0
       THEN
         (
           0.692 * (四球 - 敬遠)
           + 0.73  * 死球
           + 0.865 * (安打 - 二塁打 - 三塁打 - 本塁打)
           + 1.334 * 二塁打
           + 1.725 * 三塁打
           + 2.065 * 本塁打
         ) * 1.0
         / (打数 + 四球 - 敬遠 + 死球 + 犠飛)
  END AS wOBA,

  -- BB/K
  CASE WHEN 三振 > 0 THEN 四球 * 1.0 / 三振 END AS "BB/K",

  -- Spd（A+B+C+D)/4
  (
    (
      (( (盗塁 + 3.0) / (盗塁 + 盗塁死 + 7.0) ) - 0.4) * 20.0
    )
    +
    (
      CASE WHEN (安打 - 二塁打 - 三塁打 - 本塁打 + 四球 + 死球) > 0
           THEN (sqrt((盗塁 + 盗塁死) * 1.0 / (安打 - 二塁打 - 三塁打 - 本塁打 + 四球 + 死球)) / 0.07)
      END
    )
    +
    (
      CASE WHEN (打数 - 本塁打 - 三振) > 0
           THEN (三塁打 * 1.0 / (打数 - 本塁打 - 三振)) / 0.02 * 10.0
      END
    )
    +
    (
      CASE WHEN (安打 + 四球 + 死球 - 本塁打) > 0
           THEN (((得点 - 本塁打) * 1.0 / (安打 + 四球 + 死球 - 本塁打)) - 0.1) / 0.04
      END
    )
  ) / 4.0 AS Spd

FROM batting_2_raw;
