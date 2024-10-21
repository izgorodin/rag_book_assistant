import json

qa_pairs = [
    {
        "question": "What was the model year of the Ford Pinto involved in the accident?",
        "answer": "1972",
        "context": "In November 1971, the Grays purchased a new 1972 Ford Pinto hatchback manufactured by Ford in October 1971."
    },
    {
        "question": "Who was the executive vice president of Ford who conceived the Pinto project?",
        "answer": "Lee Iacocca",
        "context": "Mr. Iacocca, then a Ford vice president, conceived the project and was its moving force."
    },
    {
        "question": "What was the weight target for the Pinto set by Ford?",
        "answer": "2,000 pounds",
        "context": "Ford's objective was to build a car at or below 2,000 pounds to sell for no more than $2,000."
    },
    {
        "question": "On what date did the accident involving Lily Gray and Richard Grimshaw occur?",
        "answer": "May 28, 1972",
        "context": "On May 28, 1972, Mrs. Gray, accompanied by 13-year-old Richard Grimshaw, set out in the Pinto from Anaheim for Barstow to meet Mr. Gray."
    },
    {
        "question": "What was the speed of the Ford Galaxie at the moment of impact with the Pinto?",
        "answer": "28 to 37 miles per hour",
        "context": "The Galaxie had been traveling from 50 to 55 miles per hour but before the impact had been braked to a speed of from 28 to 37 miles per hour."
    },
    {
        "question": "What factors influenced the placement of the fuel tank in the Ford Pinto?",
        "answer": "Styling decisions that preceded engineering design",
        "context": "The Pinto's styling, however, required the tank to be placed behind the rear axle leaving only 9 or 10 inches of 'crush space'--far less than in any other American automobile or Ford overseas subcompact."
    },
    {
        "question": "How did Ford's crash tests reveal the Pinto's fuel system vulnerabilities?",
        "answer": "Tests showed fuel tank punctures and fuel leakage at low speeds",
        "context": "Mechanical prototypes struck from the rear with a moving barrier at 21 miles per hour caused the fuel tank to be driven forward and to be punctured, causing fuel leakage in excess of the standard prescribed by the proposed regulation."
    },
    {
        "question": "What was the estimated cost to fix the Pinto's fuel tank design, and how did Ford justify not implementing it?",
        "answer": "$6.40 per vehicle, justified by cost-benefit analysis",
        "context": "The cost to remedy the defective design of the Pinto gas tank was estimated to be $6.40 per vehicle. Ford used a cost-benefit analysis to determine that it was cheaper to pay for potential lawsuits than to fix the design."
    },
    {
        "question": "How did the court define 'malice' in the context of the Ford Pinto case?",
        "answer": "Conscious disregard for the safety of others",
        "context": "The court defined 'malice' as conduct evincing 'conscious disregard of the probability that the actor's conduct will result in injury to others.'"
    },
    {
        "question": "What was the initial punitive damage award for Grimshaw, and how was it later modified?",
        "answer": "$125 million, reduced to $3.5 million",
        "context": "Grimshaw was awarded $2,516,000 compensatory damages and $125 million punitive damages; the Grays were awarded $559,680 in compensatory damages. On Ford's motion for a new trial, Grimshaw was required to remit all but $3 1/2 million of the punitive award as a condition of denial of the motion."
    },
    {
        "question": "How did Ford's 'rush' development of the Pinto affect its safety design?",
        "answer": "Styling preceded engineering, dictating design choices",
        "context": "Pinto, however, was a rush project, so that styling preceded engineering and dictated engineering design to a greater degree than usual."
    },
    {
        "question": "What evidence suggested that Ford was aware of the Pinto's safety issues before its release?",
        "answer": "Crash test results and engineer testimonies",
        "context": "Harley Copp, a former Ford engineer and executive in charge of the crash testing program, testified that the highest level of Ford's management made the decision to go forward with the production of the Pinto, knowing that the gas tank was vulnerable to puncture and rupture at low rear impact speeds."
    },
    {
        "question": "How did the Pinto case impact the legal understanding of corporate responsibility and product liability?",
        "answer": "Established precedent for punitive damages in product liability cases",
        "context": "The case set a precedent for awarding punitive damages in product liability cases and highlighted the importance of corporate responsibility in product safety."
    },
    {
        "question": "What was the significance of the 'crush space' in the Pinto's design, and how did it compare to other vehicles?",
        "answer": "9-10 inches, less than other American cars",
        "context": "The Pinto's styling, however, required the tank to be placed behind the rear axle leaving only 9 or 10 inches of 'crush space'--far less than in any other American automobile or Ford overseas subcompact."
    },
    {
        "question": "How did Ford's internal memo about the cost-benefit analysis of safety improvements come to light during the trial?",
        "answer": "Through the testimony of Harley Copp",
        "context": "Harley Copp's testimony revealed the existence of Ford's internal cost-benefit analysis memo, which compared the cost of improvements to the potential costs of lawsuits from deaths and injuries."
    },
]
