
import re
import json

def parse_pdf_text(pdf_text):
    questions = []
    current_root_question = None
    last_question_id = None

    # Split the PDF text into lines for easier processing
    lines = pdf_text.split('\n')

    # Regex for identifying page numbers, footers, and mark allocations
    page_number_pattern = re.compile(r'^\s*Page\s\d+\s+of\s+\d+\s*$')
    footer_pattern = re.compile(r'^(www\.exam-mate\.com|\[Turn ove\r|© UCLES \d{4}\s+Page\s\d+\s+of\s+\d+\s*|\d{6}/\d{2}\s+Cambridge\s+IGCSE\s+–\s+Mark\s+Scheme\s+PUBLISHED\s+May/June\s+\d{4})$')
    mark_allocation_pattern = re.compile(r'\[(\d+)\]')
    # Regex for identifying page numbers, footers, and mark allocations
    page_number_pattern = re.compile(r'^\s*Page\s\d+\s+of\s+\d+\s*$', re.IGNORECASE)
    # Updated footer pattern to also catch the Cambridge IGCSE Mark Scheme header and empty lines
    footer_pattern = re.compile(r'^(www\.exam-mate\.com|\[Turn ove\r|© UCLES \d{4}\s+Page\s\d+\s+of\s+\d+\s*|\d{6}/\d{2}\s+Cambridge\s+IGCSE\s+–\s+Mark\s+Scheme\s+PUBLISHED\s+May/June\s+\d{4}|Generic Marking Principles|Maths-Specific Marking Principles(?:\s+\d+)?|Question Answer Marks Partial Marks)$', re.IGNORECASE)
    mark_allocation_pattern = re.compile(r'\[(\d+)\]')
    # Refined question ID pattern to be very strict: must start with digit, then optional groups (letter, roman/digit in parens)
    question_id_pattern = re.compile(r'^\s*(\d+)(?:([a-z]))?(?:\(([ivx]+|\d+)\))?(?:\(([a-z])\))?(?:\(([ivx]+|\d+)\))?(?![a-zA-Z0-9])') # Ensure not followed by more alphanumeric characters

    content_buffer = []
    last_question_id = None
    # current_parent_id will store the most recent ID that can be extended for sub-parts like (i), (a)
    current_parent_id = None
    parsing_questions = False # Flag to indicate when we are in the question parsing section
    
    roman_map = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6", "vii": "7", "viii": "8", "ix": "9", "x": "10"}

    for line_num, line in enumerate(lines):
        # Ignore empty lines and lines that are purely whitespace
        if not line.strip():
            continue

        # Apply layout and noise filters - these lines should always be ignored
        if page_number_pattern.search(line) or footer_pattern.search(line):
            continue

        # Clean unicode characters early
        cleaned_line = line.replace('\u00e2\u0080\u0093', '-') # En dash to hyphen
        cleaned_line = re.sub(r'\s*\uf0b7\s*', '', cleaned_line) # Remove bullet points
        cleaned_line = re.sub(r'\uf028|\uf029|\uf03d|\uf0b4|\uf02b|\uf02d|\uf0e6|\uf0f6|\uf0e7|\uf0f7|\uf0e8|\uf0f8', '', cleaned_line) # Remove other unicode/symbol characters
        cleaned_line = mark_allocation_pattern.sub('', cleaned_line).strip()
        
        if not cleaned_line:
            continue

        # After cleaning, check if we've reached the actual questions section
        if not parsing_questions and re.match(r'^\s*1\s*\(a\)\(i\)', cleaned_line):
            parsing_questions = True
            # Clear any preamble buffered content if we were buffering before questions started
            content_buffer = [] 

        if not parsing_questions:
            continue # Skip lines until we hit the first question

        match = question_id_pattern.match(cleaned_line)

        if match:
            # A new question ID is found, save the content of the previous question
            if content_buffer and last_question_id:
                questions.append({"question_id": last_question_id, "content": " ".join(content_buffer).strip()})
                content_buffer = [] 

            # Extract and normalize the new question ID parts
            parts = [p for p in match.groups() if p is not None]
            normalized_id_parts = []
            for part in parts:
                if part.isdigit():
                    normalized_id_parts.append(part)
                elif len(part) == 1 and part.isalpha():
                    normalized_id_parts.append(part)
                elif part.lower() in roman_map:
                    normalized_id_parts.append(roman_map[part.lower()])
                else:
                    normalized_id_parts.append(part)
            
            new_question_id = ".".join(normalized_id_parts)
            last_question_id = new_question_id
            current_parent_id = new_question_id # The current full ID is the new parent

            # Add the remaining part of the line (after the question ID) to the content buffer
            content_buffer.append(cleaned_line[match.end():].strip())
        else:
            # Contextual Continuity: If line starts with a sub-part (e.g., '(ii)' or '(b)'), it belongs to the ROOT QUESTION.
            # This implies extending the 'last_question_id' or 'current_parent_id'.
            sub_part_continuation_match = re.match(r'^\s*\(?([ivx]+|[a-z])\)?\s*(.*)$', cleaned_line)

            if sub_part_continuation_match and current_parent_id:
                sub_part_indicator = sub_part_continuation_match.group(1)
                content_after_sub_part = sub_part_continuation_match.group(2).strip()

                # Normalize sub-part indicator
                normalized_sub_part = roman_map.get(sub_part_indicator.lower(), sub_part_indicator)

                # Construct a potential new question ID by extending the current_parent_id
                potential_new_id = f"{current_parent_id}.{normalized_sub_part}"

                # Only create a new question entry if this is a genuinely new sub-part and has content.
                # This prevents creating empty entries or re-using the same ID for continuation lines.
                if potential_new_id != last_question_id and content_after_sub_part:
                    if content_buffer and last_question_id:
                        questions.append({"question_id": last_question_id, "content": " ".join(content_buffer).strip()})
                        content_buffer = []
                    last_question_id = potential_new_id
                    current_parent_id = potential_new_id # This new sub-part becomes the new parent for further nesting
                    content_buffer.append(content_after_sub_part)
                elif last_question_id: # If it's not a new sub-part ID, but belongs to the current question, append content
                    content_buffer.append(cleaned_line)
            elif last_question_id: # If no new ID and not a recognized sub-part, but still part of a question, append to current content
                content_buffer.append(cleaned_line)
            # If no last_question_id and not a recognized question/sub-part, it's considered preamble and discarded by parsing_questions flag

    # Add the last question to the list if there's any buffered content
    if content_buffer and last_question_id:
        questions.append({"question_id": last_question_id, "content": " ".join(content_buffer).strip()})

    # Final content cleaning (e.g., multiple spaces to single space)
    for q in questions:
        q["content"] = re.sub(r'\s+', ' ', q["content"]).strip()

    return json.dumps(questions, indent=2)

    return json.dumps(questions, indent=2)

# The PDF text content provided by the user
pdf_content = """
This document consists of 10 printed pages. 
 
© UCLES 2023 
 
[Turn ove
r
 
Cambridge IGCSE™ 
 
MATHEMATICS
 0580/41 
Paper 4 (Extended) May/June 2023 
MARK SCHEME 
Maximum Mark: 130 
 
 
Published 
 
 
This mark scheme is published as an aid to teachers and candidates, to indicate the requirements of the 
examination. It shows the basis on which Examiners were instructed to award marks. It does not indicate the 
details of the discussions that took place at an Examiners’ meeting before marking began, which would have 
considered the acceptability of alternative answers. 
 
Mark schemes should be read in conjunction with the question paper and the Principal Examiner Report for 
Teachers. 
 
Cambridge International will not enter into discussions about these mark schemes. 
 
Cambridge International is publishing the mark schemes for the May/June 2023 series for most 
Cambridge IGCSE, Cambridge International A and AS Level and Cambridge Pre-U components, and some 
Cambridge O Level components. 
 
 
 
www.exam-mate.com

0580/41 Cambridge IGCSE – Mark Scheme 
PUBLISHED 
May/June 2023
 
© UCLES 2023 Page 2 of 10 
 
Generic Marking Principles 
 
These general marking principles must be applied by all examiners when marking candidate answers. They 
should be applied alongside the specific content of the mark scheme or generic level descriptors for a question. 
Each question paper and mark scheme will also comply with these marking principles. 
 
GENERIC MARKING PRINCIPLE 1: 
 
Marks must be awarded in line with: 
 
 the specific content of the mark scheme or the generic level descriptors for the question 
 the specific skills defined in the mark scheme or in the generic level descriptors for the question 
 the standard of response required by a candidate as exemplified by the standardisation scripts. 
GENERIC MARKING PRINCIPLE 2: 
 
Marks awarded are always whole marks (not half marks, or other fractions). 
GENERIC MARKING PRINCIPLE 3: 
 
Marks must be awarded positively: 
 
 marks are awarded for correct/valid answers, as defined in the mark scheme. However, credit is given for 
valid answers which go beyond the scope of the syllabus and mark scheme, referring to your Team 
Leader as appropriate 
 marks are awarded when candidates clearly demonstrate what they know and can do 
 marks are not deducted for errors 
 marks are not deducted for omissions 
 answers should only be judged on the quality of spelling, punctuation and grammar when these features 
are specifically assessed by the question as indicated by the mark scheme. The meaning, however, 
should be unambiguous. 
GENERIC MARKING PRINCIPLE 4: 
 
Rules must be applied consistently, e.g. in situations where candidates have not followed instructions or in 
the application of generic level descriptors. 
GENERIC MARKING PRINCIPLE 5: 
 
Marks should be awarded using the full range of marks defined in the mark scheme for the question 
(however; the use of the full mark range may be limited according to the quality of the candidate responses 
seen). 
GENERIC MARKING PRINCIPLE 6: 
 
Marks awarded are based solely on the requirements as defined in the mark scheme. Marks should not be 
awarded with grade thresholds or grade descriptors in mind. 
 
  
www.exam-mate.com

0580/41 Cambridge IGCSE – Mark Scheme 
PUBLISHED 
May/June 2023
 
© UCLES 2023 Page 3 of 10 
 
Maths-Specific Marking Principles 
1 Unless a particular method has been specified in the question, full marks may be awarded for any correct 
method. However, if a calculation is required then no marks will be awarded for a scale drawing. 
2 Unless specified in the question, answers may be given as fractions, decimals or in standard form. Ignore 
superfluous zeros, provided that the degree of accuracy is not affected. 
3 Allow alternative conventions for notation if used consistently throughout the paper, e.g. commas being 
used as decimal points. 
4 Unless otherwise indicated, marks once gained cannot subsequently be lost, e.g. wrong working 
following a correct form of answer is ignored (isw). 
5 Where a candidate has misread a number in the question and used that value consistently throughout, 
provided that number does not alter the difficulty or the method required, award all marks earned and 
deduct just 1 mark for the misread. 
6 Recovery within working is allowed, e.g. a notation error in the working where the following line of 
working makes the candidate’s intent clear. 
 
 
Abbreviations 
cao correct answer only 
dep       dependent       
FT follow through after error 
isw ignore subsequent working 
oe  or equivalent 
SC  Special Case 
nfww    not from wrong working 
soi  seen or implied 
 
  
www.exam-mate.com

0580/41 Cambridge IGCSE – Mark Scheme 
PUBLISHED 
May/June 2023
 
© UCLES 2023 Page 4 of 10 
 
Question Answer Marks Partial Marks 
1(a)(i)         600         2 
M1 for 
1250
12    9    4


 k where k = 1, 4, 9, 12 oe 
1(a)(ii)        80        2    M1 for 1250 × 64 [÷ 1000] 
1(a)(iii)       60       2 
M1 for 
10
154
100

  


x
 oe 
1(a)(iv)       1000       2    M1 for 1250 – (1250 ÷ 5) oe 
or B1 for 250 
1(b)(i)        3.52        2    M1 for [10 –] 12 × 0.54 
or B1 for 6.48 
1(b)(ii)        0.08        3    B2    for 0.077[4...] 
 
or M1 for 0.51 ÷ 0.826  
 
If 0 or 1 scored award instead SC2 for 0.93 final 
answer 
OR  
If 0 scored SC1 for 0.06 as answe
r 
2(a) 
[sin =]
1
2
145
6.4   5.7   15
 
M2 
M1 for 145 = 
1
2
6.4   5.7   sin15 x
oe  
 
or for 
1
6.415    145
2
h
 and 
sin
5.7

h
x
  
              32.0[0]              A1 If M0, SC1 for 145 =
 0.5   6.4   5.732   15  sin
 oe 
2(b) 3.4[0] or 3.402 to 3.403 nfww 3 
M2 for 

22
6.45.72   6.4   5.7   cos  32
 
OR 
M1 for 

22
6.45.72   6.4   5.7   cos  32 
A1 for 11.6 or 11.57 to 11.58  
2(c) 3.02 or 3.020 to 3.021 3 
M2 for 

sin  32
5.7

x
 
22
80502   80   50   cos 75
 
or M1 for recognition that the line from E is 
perpendicular to AB e.g. right angle seen or 
1
6.4
2
h
  
www.exam-mate.com

0580/41 Cambridge IGCSE – Mark Scheme 
PUBLISHED 
May/June 2023
 
© UCLES 2023 Page 5 of 10 
 
Question Answer Marks Partial Marks 
2(d) 10.8 or 10.9 or 10.84 to 10.85...  4 
M3 for [sin =]
22
155.7
their(c)
 
or 


22
 
tan
5.732)15


their
cos
(c)
  
 
or M2 for 
22
155.7
 or 

2
2
5.7   cos 3215
oe 
 
or M1 for recognition of correct angle 
2(e)          136          or          136.0...          3 
M2 for 
1000
938   145
1000000

oe 
o
r M1 for figs 136 or 13601 
3(a)(i)         55.87         4        
M1 for midpoints soi  
 
M1 for use of 

fm
 where m is in the correct 
interval including boundaries  
 
M1 (dep on 2nd M1) for 

fm
÷1000 
3(a)(ii) 
177
500
 cao 
2 
M1 for 
154    200
1000

 oe 
3(b)(i)        25000        1        
3(b)(ii) 
2.473
4
10  
1        
3(c)(i) 166 650 or 165816 nfww 3    M2    for (500 + 5) × ‘320 to 340’  
or ‘500 to 510’ × (320 + 10) 
 
or M1 for 500 
 5 or 500  5 or 320 
10
 or 
320 
10
 
 
Alternative method 
M2 for 504 × ‘320 to 340’  
or ‘500 to 510’ × 329 
 
or M1 for 504 or 329 
3(c)(ii) 285 or 286 nfww 2 
M1 for 800 
10
 
4(a)(i)         96         2 
M1 for 
1
24   8
2

 
4(a)(ii)        18.4        or        18.43...        2 
M1 for 

8
tan
24
x
oe 
www.exam-mate.com

0580/41 Cambridge IGCSE – Mark Scheme 
PUBLISHED 
May/June 2023
 
© UCLES 2023 Page 6 of 10 
 
Question Answer Marks Partial Marks 
4(b) 622 or 622.0 to 622.1.... 2 
M1 for 
2
1
[]  611
2
 
 or
2
1
6[ 11]
2
  
 
4(c)(i) 246 or 246.2 to 246.3... 5 
M4 for 
2
270
15   20    4   44
360
 
 oe 
OR 
M2 for 
2
270
4
360

 oe 
or M1 for 
2
4k
, where 
1k
 
 
M1 for
15   20
 or 44 oe 
4(c)(ii) 80.8 or 80.9 or 80.84 to 80.85... 3        
M1 for 
15    20   11   16
 oe 
M1 for 
3
24
4

oe 
5(a)(i)(a)      25      1        
5(a)(i)(b)      17      to      18      1        
5(a)(i)(c)      12      2    B1    for 148 seen  
5(a)(i)(d)      30      2    B1    for 104 seen  
5(a)(ii)(a) correct diagram or correct for 
their median and LQ 
3    B1    for whiskers at 1 and at 70 
 
B1 for with median and LQ at their (a)(i)(a) and 
(a)(i)(b) 
 
B1 for UQ at 34 
Maximum 2 marks if diagram incorrect 
If 0 scored SC1 for their 5 correct ages plotted 
5(a)(ii)(b)      50      1        
5(b)          correct          histogram          3    B1    for each correct block 
width 10 height 3.7  
width 20 height 1.2  
width 30 height 2 
 
If 0 scored SC1 for correct frequency densities 
3.7, 1.2, 2 oe 
www.exam-mate.com

0580/41 Cambridge IGCSE – Mark Scheme 
PUBLISHED 
May/June 2023
 
© UCLES 2023 Page 7 of 10 
 
Question Answer Marks Partial Marks 
6(a)          (5,          2)          
(2, − 2) 
4    B3    for 3 correct values or answers for C and D 
reversed or correct coordinates given on diagram 
wrongly labelled 
or B2 for one correct coordinate pair correctly 
labelled 
or M2 for A,B,C and D correctly plotted  
or M1 for A and B correctly plotted 
 
If 0 or 1 scored instead award SC2  
for answers (–3, 8) and (–6, 4) 
or answers (1.5,1.5) and (–2.5, 4.5)  
6(b)(i) (2.5, 3.5) oe 2    B1    for each 
6(b)(ii)        7.07        or        7.071...        3 
M2 for 

22
61  43    
 oe 
or M1 for 

61 or 

43 oe 
6(b)(iii) 
1
7
 
2 
M1 for 
43
61


oe 
6(b)(iv) 
12
77
yx
 or 72yx oe 
final answer 
3    M1 for gradient = their (iii)  
 
M1dep for substituting (2, 0) in a linear 
equation with their m 
allow if (2,0) satisfies y=(their(b)(iii) 
gradient)x+c  
7(a)(i) 

33    1 3    1yyfinal answer 
3 
B2 for 

9331yy or 
 
319 3yy  
or 
or M1 for 

2
391yor [...]

3131yy 
if 0 scored SC1 for an otherwise correctly 
completely factorised expression but with 
fractions within the brackets
 
7(a)(ii) 

2pmkfinal answer 
2 
M1 for 

2  mk  pmk 
 or 

22 mpkp 
7(b) 
1
2

 oe nfww 
5 
B4 
84x
 oe nfww 
 
or B3 for 

2
85 
1
11



xx
xx
 or better 
 
OR 
B2 
2
85xx
 
or M1 for 

1161xxx
 or better 
B1 

11xx
as full denominator or on the 
right hand side 
www.exam-mate.com

0580/41 Cambridge IGCSE – Mark Scheme 
PUBLISHED 
May/June 2023
 
© UCLES 2023 Page 8 of 10 
 
Question Answer Marks Partial Marks 
7(c) 
   
2
33442
24
        

oe 
 
or 
2
332
884




 oe 
M2 
M1 for 
 
2
3442  
 
 
or for 


3
24
 q
 or 


3
24
 q
  
 
or for [4]
2
3
8




x
 
 −0.43 and 1.18 final ans cao  
A2 
B1 for each 
 
SC1 for −0.4 ,–0.42 or −0.425....  
and 1.2 or 1.17 or 1.175.... 
or answers 0.43 and 1.18  
or −0.43 an
d 1.18 seen in working 
7(d) 
4
1


m
k
pm
 or 
4
1



m
k
pm
  
 
final answer 
4        
 
 
 
M1 for clearing fractions 
 
M1 for collecting terms in k 
M1 for factorising 
M1 for dividing by bracket 
Maximum 3 marks if answer incorrect 
8(a) 7yoe 
14xyoe 
2
3
yx
 oe 
3    B1    for each 
8(b) 
4x
 solid 
7y solid 
14xy dashed 
2
3
yx
 dashed 
M4    B1    for each 
 correct shading everywhere but 
region R 
 
 
A2    M1dep    (dependent on M4 or B1B1B1B0 where 
the only error is wrong use of solid/dashed lines) 
for shading the correct side of 3 of the 4 lines. 
R 
 
 
 
 
www.exam-mate.com

0580/41 Cambridge IGCSE – Mark Scheme 
PUBLISHED 
May/June 2023
 
© UCLES 2023 Page 9 of 10 
 
Question Answer Marks Partial Marks 
8(c) 4 dresses and 3 shirts 1        
8(d)          106          2 
M1 for 
106xyevaluated for (x, y) in their 
region R 
or B1 for (7, 6) 
 
After 0 scored, SC1 for answer 112 or 116 
9(a)(i) r, l, t, e, a 1                                                                                                                                                                      
9(a)(ii)        2        1                                                                                                                                                                      
9(b) 
 
1        
 
 
1        
9(c)(i)         Fully         correct         
 
3    B2    for 7, 6, or 5 sections correct  
or B1 for 4, 3 or 2 sections correct 
9(c)(ii)        5                1FT strict FT from their diagram 
10(a)(i) −7 1        
10(a)(ii) 
5
2
x
 oe final answer 
2    M1    for correct first step e.g. 
25xy or 
25xy  
or 
5
22

y
x
 
10(a)(iii) 
32
211880xxxfinal answer 
4 
M1 for 

4x

25x

4xoe 
B2 for 
322 2
288520203280xxxx  x x x 
or for simplified 4 term expression of the correct 
form with 3 terms correct in final answer 
or B1 for 3 terms correct out of 4 from 
2
4416xxx or 
2
28520xxx 
 
 
8 
 
 
 
 
1 
2 
3 
4 
5 
6
7 
9 
10 
11 
12 
www.exam-mate.com
"""

if __name__ == "__main__":
    parsed_json = parse_pdf_text(pdf_content)
    print(parsed_json)
