combined_diagnoses_query = """
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
    sepsis AS (
        SELECT vo.visit_occurrence_id,
            p.person_id,
            p.year_of_birth,
            p.gender_source_value,
            c.concept_id,
            c.concept_name,
            c.concept_code,
            1 AS sepsis_at_admission,
            ROW_NUMBER() OVER(PARTITION BY vo.visit_occurrence_id ORDER BY co.condition_start_date DESC) AS rownum
        FROM person p
        JOIN visit_occurrence vo
            ON p.person_id = vo.person_id
        JOIN condition_occurrence co
            ON vo.visit_occurrence_id = co.visit_occurrence_id
        JOIN concept c
            ON co.condition_concept_id = c.concept_id
        WHERE c.concept_id = 132797.0 -- Sepsis concept_id
    ), sepsis_antibiotics AS (
        SELECT
            vo.visit_occurrence_id,
            CASE
                WHEN COUNT(*) > 0 THEN 1
                ELSE 0
            END AS sepsis_antibiotics_bool,
            STRING_AGG(DISTINCT c.concept_name, '; ') AS sepsis_antibiotics_given
        FROM drug_exposure de
        JOIN visit_occurrence vo
            ON de.visit_occurrence_id = vo.visit_occurrence_id
        JOIN concept c
            ON de.drug_concept_id = c.concept_id
        WHERE c.concept_id IN (
            -- List of concept_ids corresponding to antibiotics
            36258242.0, -- Amikacin
            41114247.0, -- Amoxicillin Injectable Solution [CLAMOXYL]
            40072606.0, -- penicillin G Injectable Solution
            40730323.0, -- Ceftazidime Injectable Solution [Fortum]
            40737711.0, -- Ciprofloxacin Injectable Solution
            40754936.0, --Rifampicine (Rifadin)
            35154887.0, --Clindamycine (Dalacin)
            36257100.0, --Tobramycin 40 MG/ML Injectable Solution [Obracin]
            --43295826.0, --Vancomycine -> prophylaxis for valve surgery
            43023020.0, --Cilastatin 500 MG / Imipenem 500 MG Injectable Solution [TIENAM]
            44125439.0, --Doxycycline (Vibramycine)
            --42479695.0, --Metronidazol (Flagyl) -> often used for GI surgical prophylaxis
            --36783613.0, --Erythromycine (Erythrocine) -> often used for gastroparesis
            40040317.0, --Flucloxacilline (Stafoxil/Floxapen)
            40846188.0, --Fluconazol (Diflucan)
            35761065.0, --Ganciclovir (Cymevene)
            36883151.0, --Flucytosine (Ancotil)
            44086327.0, --Gentamicine (Garamycin)
            35606304.0, --Foscarnet trinatrium (Foscavir)
            41239210.0, --Amfotericine B (Fungizone)
            40928582.0, --Meropenem (Meronem)
            43744909.0, --Myambutol (ethambutol)
            1760659.0, --Kinine dihydrocloride
            --783871.0, --Immunoglobuline (Nanogam) -> not anbiotic
            --40902269.0, --Co-Trimoxazol (Bactrimel) -> often prophylactic (unless high dose)
            40748259.0, --Voriconazol(VFEND)
            --21091196.0, --Amoxicilline/Clavulaanzuur (Augmentin) -> often used for ENT surgical prophylaxis
            40740535.0, --Aztreonam (Azactam)
            40023477.0, --Chlooramfenicol
            --40988255.0, --Fusidinezuur (Fucidin) -> prophylaxis
            40077149.0, --Piperacilline (Pipcil)
            40731671.0, --Ceftriaxon (Rocephin)
            --40020952.0, --Cefuroxim (Zinacef) -> often used for GI/transplant surgical prophylaxis
            --40736877.0, --Cefazoline (Kefzol) -> prophylaxis for cardiac surgery
            40735073.0, --Caspofungine
            40052913.0, --Itraconazol (Trisporal)
            --35417660.0, --Tetanusimmunoglobuline -> prophylaxis/not antibiotic
            36886030.0, --Levofloxacine (Tavanic)
            35604141.0, --Amfotericine B lipidencomplex  (Abelcet)
            35750082.0, --Ecalta (Anidulafungine)
            40747347.0, --Amfotericine B in liposomen (Ambisome )
            36896811.0, --Linezolid (Zyvoxid)
            40734887.0, --Tigecycline (Tygacil)
            40736859.0, --Daptomycine (Cubicin)
            40241491.0 --Colistine
        )
        AND de.drug_exposure_start_datetime < TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
        GROUP BY vo.visit_occurrence_id
    ), other_antibiotics AS (
        SELECT
            vo.visit_occurrence_id,
            CASE
                WHEN COUNT(*) > 0 THEN 1
                ELSE 0
            END AS other_antibiotics_bool,
            STRING_AGG(DISTINCT c.concept_name, '; ') AS other_antibiotics_given
        FROM drug_exposure de
        JOIN visit_occurrence vo
            ON de.visit_occurrence_id = vo.visit_occurrence_id
        JOIN concept c
            ON de.drug_concept_id = c.concept_id
        WHERE c.concept_id IN (
            -- List of concept_ids corresponding to prophylactic antibiotics
            43295826.0, --Vancomycine -> prophylaxis for valve surgery
            42479695.0, --Metronidazol (Flagyl) -> often used for GI surgical prophylaxis
            40902269.0, --Co-Trimoxazol (Bactrimel) -> often prophylactic (unless high dose)
            21091196.0, --Amoxicilline/Clavulaanzuur (Augmentin) -> often used for ENT surgical prophylaxis
            40020952.0, --Cefuroxim (Zinacef) -> often used for GI surgical prophylaxis
            40736877.0 --Cefazoline (Kefzol) -> prophylaxis
        )
        AND de.drug_exposure_start_datetime < TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
        GROUP BY vo.visit_occurrence_id
    ), cultures AS (
        SELECT
            vo.visit_occurrence_id,
            CASE
                WHEN COUNT(*) > 0 THEN 1
                ELSE 0
            END AS sepsis_cultures_bool,
            STRING_AGG(DISTINCT c.concept_name, '; ') AS sepsis_cultures_drawn
        FROM filtered_measurement m
        JOIN visit_occurrence vo
            ON m.visit_occurrence_id = vo.visit_occurrence_id
        JOIN concept c
            ON m.measurement_concept_id = c.concept_id
        WHERE c.concept_id IN (
            -- List of concept_ids for culture measurements
            --4015189.0, --Sputumkweek afnemen -> often used routinely
            --4024509.0, --Urinekweek afnemen
            --8588, --MRSA kweken afnemen
            1761466.0, --Bloedkweken afnemen
            3009986.0, --Cathetertipkweek afnemen
            --3008334.0, --Drainvochtkweek afnemen
            --37392597.0, --Faeceskweek afnemen -> Clostridium
            --4189544.0, --X-Kweek nader te bepalen
            --4098503.0, --Liquorkweek afnemen
            --37117051.0, --Neuskweek afnemen
            --3040827.0, --Perineumkweek afnemen -> often used routinely
            -3040827.0, --Rectumkweek afnemen -> often used routinely
            3003714.0, --Wondkweek afnemen
            3025037.0, --Ascitesvochtkweek afnemen
            --4024958.0, --Keelkweek afnemen -> often used routinely
            4196406.0 --Legionella sneltest (urine)
        )
        AND m.measurement_datetime < TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 6 HOUR)
        GROUP BY vo.visit_occurrence_id
    )
    SELECT
        vo.*,
        sepsis.person_id,
        sepsis.year_of_birth,
        sepsis.gender_source_value,
        sepsis.concept_id,
        sepsis.concept_name,
        sepsis.concept_code,
        sepsis.sepsis_at_admission,
        sepsis_antibiotics.sepsis_antibiotics_bool,
        sepsis_antibiotics.sepsis_antibiotics_given,
        other_antibiotics.other_antibiotics_bool,
        other_antibiotics.other_antibiotics_given,
        cultures.sepsis_cultures_bool,
        cultures.sepsis_cultures_drawn
    FROM visit_occurrence vo
    LEFT JOIN sepsis ON vo.visit_occurrence_id = sepsis.visit_occurrence_id
    LEFT JOIN sepsis_antibiotics ON vo.visit_occurrence_id = sepsis_antibiotics.visit_occurrence_id
    LEFT JOIN other_antibiotics ON vo.visit_occurrence_id = other_antibiotics.visit_occurrence_id
    LEFT JOIN cultures ON vo.visit_occurrence_id = cultures.visit_occurrence_id
    WHERE sepsis.rownum = 1 OR sepsis.rownum IS NULL;
    """
