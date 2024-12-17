"""Module providing all queries related to sofa_scores.

Author: Mika Florin Rosin

query author: Laurens Biesheuvel"""


query_all_admissions = """
SELECT * FROM visit_occurrence
"""

query_sofa_respiratory = '''
WITH filtered_measurement AS (
    SELECT *
    FROM measurement
    WHERE provider_id IS NOT NULL
),
filtered_observation AS (
    SELECT *
    FROM observation
    WHERE provider_id IS NOT NULL
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
    -- measurements within 24 hours of ICU stay:
    AND TIMESTAMP_DIFF(n.measurement_datetime, a.visit_start_datetime, HOUR) <= 24
    AND TIMESTAMP_DIFF(n.measurement_datetime, a.visit_start_datetime, HOUR) >= 0
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

query_sofa_coagulation = """
WITH filtered_measurement AS (
    SELECT *
    FROM measurement
    WHERE provider_id IS NOT NULL
),
sofa_platelets AS (
    SELECT
        n.visit_occurrence_id,
        n.measurement_concept_id AS itemid,
        n.measurement_source_value AS item,
        n.value_as_number AS value,
        n.provider_id AS registeredby,
        TIMESTAMP_DIFF(n.measurement_datetime, a.visit_start_datetime, HOUR) AS time
    FROM filtered_measurement n
    LEFT JOIN visit_occurrence a ON
        n.visit_occurrence_id = a.visit_occurrence_id
    WHERE
        n.measurement_concept_id = 3007461 -- Platelets (Thrombo's (bloed))
        AND TIMESTAMP_DIFF(n.measurement_datetime, a.visit_start_datetime, HOUR) BETWEEN -0.5 AND 24
)
SELECT * FROM sofa_platelets
"""


query_sofa_liver = """
WITH filtered_measurement AS (
    SELECT *
    FROM measurement
    WHERE provider_id IS NOT NULL
),
sofa_bilirubin AS (
    SELECT
        n.visit_occurrence_id,
        n.measurement_concept_id AS itemid,
        n.measurement_source_value AS item,
        n.value_as_number AS value,
        n.provider_id AS registeredby,
        TIMESTAMP_DIFF(n.measurement_datetime, a.visit_start_datetime, HOUR) AS time
    FROM filtered_measurement n
    LEFT JOIN visit_occurrence a ON
        n.visit_occurrence_id = a.visit_occurrence_id
    WHERE
        n.measurement_concept_id IN (
            40757494, -- Bili Totaal
            3006140   -- Bilirubine (bloed)
        )
        AND TIMESTAMP_DIFF(n.measurement_datetime, a.visit_start_datetime, HOUR) BETWEEN -0.5 AND 24
)
SELECT * FROM sofa_bilirubin
"""

query_lactate_sepsis = """
WITH filtered_measurement AS (
    SELECT *
    FROM measurement
    WHERE provider_id IS NOT NULL
),
initial_lactate AS (
    SELECT
        n.visit_occurrence_id,
        n.measurement_concept_id AS itemid,
        n.measurement_source_value AS item,
        n.value_as_number AS value,
        n.provider_id AS registeredby,
    FROM filtered_measurement n
    LEFT JOIN visit_occurrence a ON
        n.visit_occurrence_id = a.visit_occurrence_id
    WHERE
        n.measurement_concept_id IN (
            3047181,
            3014111
        )
)
SELECT * FROM initial_lactate
"""


query_vasopressors_ionotropes = """
WITH filtered_measurement AS (
    SELECT
        visit_occurrence_id,
        value_as_number AS patientweight
    FROM measurement
    WHERE provider_id IS NULL
    AND measurement_concept_id IN (
        3026600, -- Body weight Estimated
        3013762, -- Body weight Measured
        3023166, -- Body weight Stated
        3025315  -- Body weight
    )
),
dosing AS (
    SELECT
        de.visit_occurrence_id,
        de.drug_concept_id AS itemid,
        c.concept_name AS item,
        TIMESTAMP_DIFF(de.drug_exposure_start_datetime, vo.visit_start_datetime, MINUTE) AS start_time,
        TIMESTAMP_DIFF(de.drug_exposure_end_datetime, vo.visit_start_datetime, MINUTE) AS stop_time,
        TIMESTAMP_DIFF(de.drug_exposure_end_datetime, de.drug_exposure_start_datetime, MINUTE) AS duration,
        -- Extract dose and rate from the sig field
        CAST(REGEXP_EXTRACT(de.sig, r'(\\d+\\.?\\d*) mg') AS FLOAT64) AS dose,
        CAST(REGEXP_EXTRACT(de.sig, r'@ (\\d+\\.?\\d*) mg/uur') AS FLOAT64) AS rate,
        'mg/uur' AS rateunit,
        fm.patientweight
    FROM drug_exposure de
    LEFT JOIN visit_occurrence vo ON de.visit_occurrence_id = vo.visit_occurrence_id
    LEFT JOIN concept c ON de.drug_concept_id = c.concept_id
    LEFT JOIN filtered_measurement fm ON de.visit_occurrence_id = fm.visit_occurrence_id
    WHERE c.concept_id IN (
            36411287, -- 50 ML Dopamine 4 MG/ML Injectable Solution
            21088391, -- 50 ML Dobutamine 5 MG/ML Injection
            19076867, -- Epinephrine 0.1 MG/ML Injectable Solution
            2907531  -- 50 ML Norepinephrine 0.2 MG/ML Injection
        )
    AND CAST(REGEXP_EXTRACT(de.sig, r'@ (\\d+\\.?\\d*) mg/uur') AS FLOAT64) > 0.1
)
SELECT
    visit_occurrence_id,
    itemid,
    item,
    duration,
    dose,
    rate,
    rateunit,
    start_time,
    stop_time,
    patientweight,
    CASE
        -- recalculate the dose to µg/kg/min ('gamma')
        WHEN rateunit = 'mg/uur' THEN (rate * 1000) / patientweight / 60 -- convert mg/hour to µg/kg/min
        ELSE NULL -- Placeholder for other conversions if necessary
    END AS gamma
FROM dosing
WHERE
    -- medication given within 24 hours of ICU stay:
    start_time <= 1440 AND stop_time >= 0 -- 24 * 60 minutes
ORDER BY visit_occurrence_id, start_time
"""


query_blood_pressure = """
WITH filtered_measurement AS (
    SELECT *
    FROM measurement
    WHERE (provider_id IS NULL OR provider_id IS NOT NULL)
    AND measurement_concept_id IN (
        21490852, -- ABP gemiddeld
        21492241, -- Niet invasieve bloeddruk gemiddeld
        21490852  -- ABP gemiddeld II
    )
)
SELECT
    m.visit_occurrence_id,
    m.measurement_concept_id AS itemid,
    m.measurement_source_value AS item,
    m.value_as_number AS value,
    CASE
        WHEN m.provider_id IS NOT NULL THEN TRUE
        ELSE FALSE
    END as validated,
    TIMESTAMP_DIFF(m.measurement_datetime, vo.visit_start_datetime, MINUTE) AS time
FROM filtered_measurement m
LEFT JOIN visit_occurrence vo ON m.visit_occurrence_id = vo.visit_occurrence_id
WHERE
    TIMESTAMP_DIFF(m.measurement_datetime, vo.visit_start_datetime, MINUTE) <= 1440 -- 24 * 60 minutes
ORDER BY m.visit_occurrence_id, time
"""

query_sofa_cns = """
WITH filtered_measurement AS (
    SELECT *
    FROM measurement
    WHERE provider_id IS NOT NULL
    AND measurement_concept_id IN (
        3016335, -- Glasgow coma score eye opening
        3026549, -- Glasgow coma score motor at First encounter
        3009094  -- Glasgow coma score verbal
    )
),
gcs_components AS (
    SELECT
        m.visit_occurrence_id,
        CASE m.measurement_concept_id
            WHEN 3016335 THEN
                CASE m.value_as_concept_id
                    WHEN 45877537 THEN 1 -- No eye opening
                    WHEN 45883351 THEN 2 -- Eye opening to pain
                    WHEN 45880465 THEN 3 -- Eye opening to verbal command
                    WHEN 45880466 THEN 4 -- Eyes open spontaneously
                    ELSE NULL
                END
            ELSE NULL
        END AS eyes_score,
        CASE m.measurement_concept_id
            WHEN 3026549 THEN
                CASE m.value_as_concept_id
                    WHEN 45878992 THEN 1 -- No motor response
                    WHEN 45878993 THEN 2 -- Extension to pain
                    WHEN 45879885 THEN 3 -- Flexion to pain
                    WHEN 45882047 THEN 4 -- Withdrawal from pain
                    WHEN 45880468 THEN 5 -- Obeys commands
                    ELSE NULL
                END
            ELSE NULL
        END AS motor_score,
        CASE m.measurement_concept_id
            WHEN 3009094 THEN
                CASE m.value_as_concept_id
                    WHEN 36311192 THEN 1 -- Intubated
                    WHEN 45877384 THEN 2 -- No verbal response
                    WHEN 45883352 THEN 3 -- Incomprehensible sounds
                    WHEN 45877601 THEN 4 -- Inappropriate words
                    WHEN 45883906 THEN 5 -- Confused
                    WHEN 45877602 THEN 6 -- Oriented
                    ELSE NULL
                END
            ELSE NULL
        END AS verbal_score,
        m.provider_id,
        TIMESTAMP_DIFF(m.measurement_datetime, vo.visit_start_datetime, MINUTE) AS time
    FROM filtered_measurement m
    LEFT JOIN visit_occurrence vo ON m.visit_occurrence_id = vo.visit_occurrence_id
    WHERE TIMESTAMP_DIFF(m.measurement_datetime, vo.visit_start_datetime, MINUTE) <= 1440 -- 24 * 60 minutes
),
gcs AS (
    SELECT
        *,
        COALESCE(eyes_score, 0) + COALESCE(motor_score, 0) + (
            CASE
                WHEN COALESCE(verbal_score, 0) < 1 THEN 1
                ELSE COALESCE(verbal_score, 0)
            END
        ) AS gcs_score
    FROM gcs_components
),
gcs_prioritized AS (
    SELECT *,
        ROW_NUMBER() OVER(
            PARTITION BY visit_occurrence_id
            ORDER BY time DESC, gcs_score DESC
        ) AS priority
    FROM gcs
)
SELECT *
FROM gcs_prioritized
WHERE priority = 1
ORDER BY visit_occurrence_id, gcs_score DESC
"""


query_sofa_renal = """
WITH filtered_measurement AS (
    SELECT *
    FROM measurement
    WHERE provider_id IS NOT NULL
    AND measurement_concept_id IN (
        3014315, -- Urine
        3007123, -- UrineSpontaan
        21491173, -- Nefrodrain
        3014315  -- UrineSplint
    )
)
SELECT
    m.visit_occurrence_id,
    m.measurement_concept_id AS itemid,
    m.measurement_source_value AS item,
    m.value_as_number AS value,
    m.provider_id AS registeredby,
    TIMESTAMP_DIFF(m.measurement_datetime, vo.visit_start_datetime, MINUTE) AS time
FROM filtered_measurement m
LEFT JOIN visit_occurrence vo ON m.visit_occurrence_id = vo.visit_occurrence_id
WHERE TIMESTAMP_DIFF(m.measurement_datetime, vo.visit_start_datetime, MINUTE) BETWEEN 0 AND 1440 -- within 24 hours
ORDER BY m.visit_occurrence_id, time
"""


query_sofa_creatinine = """
WITH filtered_measurement AS (
    SELECT *
    FROM measurement
    WHERE provider_id IS NOT NULL
    AND measurement_concept_id = 3020564 -- Kreatinine
),
baseline AS (
    SELECT
        m.visit_occurrence_id,
        MIN(m.value_as_number) AS baseline_creatinine
    FROM filtered_measurement m
    LEFT JOIN visit_occurrence vo ON m.visit_occurrence_id = vo.visit_occurrence_id
    WHERE
        TIMESTAMP_DIFF(m.measurement_datetime, vo.visit_start_datetime, HOUR) > -8760 -- up to 1 year before admission
        AND TIMESTAMP_DIFF(m.measurement_datetime, vo.visit_start_datetime, HOUR) < 24 -- within 24 hours after admission
    GROUP BY m.visit_occurrence_id
),
max_creat AS (
    SELECT
        m.visit_occurrence_id,
        MAX(m.value_as_number) AS max_creatinine_7days
    FROM filtered_measurement m
    LEFT JOIN visit_occurrence vo ON m.visit_occurrence_id = vo.visit_occurrence_id
    WHERE
        TIMESTAMP_DIFF(m.measurement_datetime, vo.visit_start_datetime, HOUR) BETWEEN 0 AND 168 -- within 7 days of admission
    GROUP BY m.visit_occurrence_id
)
SELECT
    m.visit_occurrence_id,
    m.measurement_concept_id AS itemid,
    m.measurement_source_value AS item,
    m.value_as_number AS value,
    m.provider_id AS registeredby,
    TIMESTAMP_DIFF(m.measurement_datetime, vo.visit_start_datetime, MINUTE) AS time,
    b.baseline_creatinine,
    mx.max_creatinine_7days,
    CASE
        -- AKI definition: 3 fold increase:
        WHEN b.baseline_creatinine > 0 AND mx.max_creatinine_7days / b.baseline_creatinine > 3 THEN TRUE
        -- AKI definition: increase to >= 354 umol/l AND at least 44 umol/l increase:
        WHEN mx.max_creatinine_7days >= 354 AND mx.max_creatinine_7days - b.baseline_creatinine >= 44 THEN TRUE
        ELSE FALSE
    END AS acute_renal_failure
FROM filtered_measurement m
LEFT JOIN visit_occurrence vo ON m.visit_occurrence_id = vo.visit_occurrence_id
LEFT JOIN baseline b ON m.visit_occurrence_id = b.visit_occurrence_id
LEFT JOIN max_creat mx ON m.visit_occurrence_id = mx.visit_occurrence_id
WHERE
    TIMESTAMP_DIFF(m.measurement_datetime, vo.visit_start_datetime, MINUTE) BETWEEN -30 AND 1440 -- within 24 hours (and 30 minutes before)
ORDER BY m.visit_occurrence_id, time
"""
