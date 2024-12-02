query_demographics = '''
    SELECT p.person_id, p.year_of_birth, p.gender_source_value
    FROM person as p

    --LEFT JOIN concept c ON m.measurement_concept_id = c.concept_id

    WHERE p.person_id IN {person_ids}
    GROUP BY p.person_id, p.year_of_birth, p.gender_source_value
    '''

query_measurement = '''
SELECT
    m.person_id,
    m.measurement_concept_id,
    c.concept_name,
    measurement_datetime,
    value_as_number,
    m.visit_occurrence_id
FROM measurement as m
LEFT JOIN concept c
    ON m.measurement_concept_id = c.concept_id
WHERE
    m.person_id IN {person_ids}
    AND m.provider_id IS NULL
    AND measurement_concept_id IN (
        3026600, -- Body weight (stated/estimated/measured)
        3013762,
        3023166,
        3025315
    )
GROUP BY
    m.person_id,
    m.measurement_concept_id,
    c.concept_name,
    measurement_datetime,
    value_as_number,
    m.visit_occurrence_id

UNION ALL

SELECT
    m.person_id,
    m.measurement_concept_id,
    c.concept_name,
    measurement_datetime,
    value_as_number,
    m.visit_occurrence_id
FROM measurement as m
LEFT JOIN concept c
    ON m.measurement_concept_id = c.concept_id
WHERE
    m.person_id IN {person_ids}
    AND (m.provider_id BETWEEN 27 AND 61 OR m.provider_id IS NULL)
    AND c.concept_id IN {concept_ids}
GROUP BY
    m.person_id,
    m.measurement_concept_id,
    c.concept_name,
    measurement_datetime,
    value_as_number,
    m.visit_occurrence_id
'''