from typing import List


class Students:
    def __init__(self, names:List[str], grades: List[float]):
        self.names = names
        self.grades = grades
    
    def __len__(self):
        return len(self.names)
    
    def __add__(self, other_students):
        return Students(self.names + other_students.names,
                        self.grades + other_students.grades)
    
    def __repr__(self):
        return f'names: {self.names}, grades: {self.grades}'
