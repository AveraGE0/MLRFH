query_demographics = '''
    SELECT vo.visit_occurrence_id, DATE_DIFF(DATE(vo.visit_start_datetime), DATE(CONCAT(p.year_of_birth, '-01-01')), YEAR) AS age_at_visit, p.gender_source_value
    FROM visit_occurrence AS vo

    LEFT JOIN person as p on vo.person_id = p.person_id

    WHERE vo.visit_occurrence_id IN {visit_occurrence_ids}
    GROUP BY vo.visit_occurrence_id, age_at_visit, p.gender_source_value
    '''

query_measurement = '''
SELECT
    m.measurement_concept_id,
    c.concept_name,
    measurement_datetime,
    value_as_number,
    m.visit_occurrence_id
FROM measurement as m
LEFT JOIN concept c
    ON m.measurement_concept_id = c.concept_id
WHERE
    m.visit_occurrence_id IN {visit_occurrence_ids}
    AND m.provider_id IS NULL
    AND measurement_concept_id IN {weight_temp_ids}
GROUP BY
    m.measurement_concept_id,
    c.concept_name,
    measurement_datetime,
    value_as_number,
    m.visit_occurrence_id

UNION ALL

SELECT
    m.measurement_concept_id,
    c.concept_name,
    measurement_datetime,
    value_as_number,
    m.visit_occurrence_id
FROM measurement as m
LEFT JOIN concept c
    ON m.measurement_concept_id = c.concept_id
WHERE
    m.visit_occurrence_id IN {visit_occurrence_ids}
    AND m.provider_id BETWEEN 27 AND 61
    AND c.concept_id IN {concept_ids}
GROUP BY
    m.measurement_concept_id,
    c.concept_name,
    measurement_datetime,
    value_as_number,
    m.visit_occurrence_id
'''

query_ventilation = '''
WITH filtered_measurement AS (
    SELECT *
    FROM measurement
    WHERE provider_id IS NOT NULL
   AND visit_occurrence_id IN {visit_occurrence_ids}
),
filtered_observation AS (
    SELECT *
    FROM observation
    WHERE provider_id IS NOT NULL
    AND visit_occurrence_id IN {visit_occurrence_ids}
),
fio2_table AS (
    SELECT
        n.visit_occurrence_id,
        n.measurement_datetime,
        l.value_as_number,
        l.observation_concept_id AS O2_device,
        CASE
            WHEN n.measurement_concept_id IN (
                -- FiO2 settings on respiratory support
                3025408, -- O2 concentratie --measurement by Servo-i/Servo-U ventilator
                3024882  -- SET %O2
            ) THEN TRUE
            ELSE FALSE
        END AS ventilatory_support,
        CASE
            WHEN n.measurement_concept_id IN (
                -- FiO2 settings on respiratory support
                3025408, -- O2 concentratie --measurement by Servo-i/Servo-U ventilator
                3024882  -- SET %O2
            ) THEN
                CASE
                    WHEN NOT n.value_as_number IS NULL THEN n.value_as_number -- use the settings
                    ELSE 0.21
                END
            ELSE -- estimate the FiO2
                CASE
                    -- Updated nasal cannula concept_ids
                    WHEN l.observation_concept_id IN (
                        4224038 -- nasal catheter
                    ) THEN
                        CASE
                            WHEN n.value_as_number >= 1 AND n.value_as_number < 2 THEN 0.22
                            WHEN n.value_as_number >= 2 AND n.value_as_number < 3 THEN 0.25
                            WHEN n.value_as_number >= 3 AND n.value_as_number < 4 THEN 0.27
                            WHEN n.value_as_number >= 4 AND n.value_as_number < 5 THEN 0.30
                            WHEN n.value_as_number >= 5 THEN 0.35
                            ELSE 0.21
                        END
                    WHEN l.observation_concept_id IN (
                        45759142 -- Oxygen administration nasal catheter (Diep Nasaal)
                    ) THEN
                        CASE
                            WHEN n.value_as_number >= 1 AND n.value_as_number < 2 THEN 0.22 -- not defined by NICE
                            WHEN n.value_as_number >= 2 AND n.value_as_number < 3 THEN 0.25
                            WHEN n.value_as_number >= 3 AND n.value_as_number < 4 THEN 0.27
                            WHEN n.value_as_number >= 4 AND n.value_as_number < 5 THEN 0.30
                            WHEN n.value_as_number >= 5 AND n.value_as_number < 6 THEN 0.35
                            WHEN n.value_as_number >= 6 AND n.value_as_number < 7 THEN 0.40
                            WHEN n.value_as_number >= 7 AND n.value_as_number < 8 THEN 0.45
                            WHEN n.value_as_number >= 8 THEN 0.50
                            ELSE 0.21
                        END
                    WHEN l.observation_concept_id IN (
                        4192506, -- Waterset
                        4044008, -- Trach.stoma
                        4160626, -- Ambu
                        4235033, -- Guedel
                        4208623, -- DL-tube
                        4165535, -- CPAP
                        4145528  -- Non-Rebreathing masker
                    ) THEN
                        CASE
                            WHEN n.value_as_number >= 6 AND n.value_as_number < 7 THEN 0.60
                            WHEN n.value_as_number >= 7 AND n.value_as_number < 8 THEN 0.70
                            WHEN n.value_as_number >= 8 AND n.value_as_number < 9 THEN 0.80
                            WHEN n.value_as_number >= 9 AND n.value_as_number < 10 THEN 0.85
                            WHEN n.value_as_number >= 10 THEN 0.90
                            ELSE 0.21
                        END
                    WHEN l.observation_concept_id IN (
                        45879331 -- B.Lucht
                    ) THEN 0.21
                ELSE 0.21
            END
        END AS fio2
    FROM filtered_measurement n
    LEFT JOIN visit_occurrence a ON
        n.visit_occurrence_id = a.visit_occurrence_id
    LEFT JOIN filtered_observation l ON
        n.visit_occurrence_id = l.visit_occurrence_id AND
        n.measurement_datetime = l.observation_datetime AND
        l.observation_concept_id = 45769206 -- Toedieningsweg (Oxygen device)
    WHERE
        n.measurement_concept_id IN (
            --Oxygen Flow settings without respiratory support
            3014080, -- O2 l/min
            3005629, --Zuurstof toediening
            3025408, -- O2 concentratie --measurement by Servo-i/Servo-U ventilator
            3024882  -- SET %O2
        )
    AND n.value_as_number > 0 -- ignore stand by values from Evita ventilator
),
oxygenation AS (
    SELECT
        pao2.visit_occurrence_id,
        CASE pao2.unit_concept_id
            WHEN 152 THEN pao2.value_as_number * 7.50061683 -- Conversion: kPa to mmHg
            ELSE pao2.value_as_number
        END AS pao2,
        f.value_as_number AS specimen_source,
        CASE
            WHEN pao2.provider_id IS NOT NULL THEN TRUE
            ELSE FALSE
        END AS manual_entry,
        TIMESTAMP_DIFF(pao2.measurement_datetime, a.visit_start_datetime, HOUR) AS time,
        fio2_table.fio2,
        fio2_table.ventilatory_support,
        TIMESTAMP_DIFF(fio2_table.measurement_datetime, pao2.measurement_datetime, MINUTE) AS FiO2_time_difference,
        fio2_table.measurement_datetime as measurement_datetime,
        ROW_NUMBER() OVER(
            PARTITION BY pao2.visit_occurrence_id, pao2.measurement_datetime
            ORDER BY ABS(TIMESTAMP_DIFF(fio2_table.measurement_datetime, pao2.measurement_datetime, MINUTE))
        ) AS priority -- give priority to nearest FiO2 measurement
    FROM filtered_measurement pao2
    LEFT JOIN visit_occurrence a ON
        pao2.visit_occurrence_id = a.visit_occurrence_id
    LEFT JOIN filtered_measurement f ON
        pao2.visit_occurrence_id = f.visit_occurrence_id AND
        pao2.measurement_datetime = f.measurement_datetime AND
        f.measurement_concept_id = 3053213 -- Afname (bloed): source of specimen
    LEFT JOIN filtered_measurement paco2 ON
        pao2.visit_occurrence_id = paco2.visit_occurrence_id AND
        pao2.measurement_datetime = paco2.measurement_datetime AND
        paco2.measurement_concept_id IN (
            3013290, -- PCO2
            3013290, -- pCO2 (bloed)
            3013290  -- PCO2 (bloed) - kPa
        )
    LEFT JOIN fio2_table ON
        pao2.visit_occurrence_id = fio2_table.visit_occurrence_id
    WHERE
        pao2.measurement_concept_id IN (
            3027315, -- PO2
            3027315, -- PO2 (bloed)
            3027315  -- PO2 (bloed) - kPa
        )
)
SELECT * FROM oxygenation
WHERE priority = 1
'''