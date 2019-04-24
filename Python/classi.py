#programmazione procedurale
# definiamo due funzioni per calcolare area e perimetro
def calc_rectangle_area(base, height):
    """Calculate and return the area of a rectangle."""
    return base * height
def calc_rectangle_perimeter(base, height):
    """Calculate and return the perimeter of a rectangle."""
    return (base + height) * 2

# programmazione ad oggetti
#i metodi devono definire un parametro aggiuntivo che per convenzione è chiamato self
#e classi supportano anche diversi metodi “speciali” che sono identificati dalla presenza di due underscore prima e dopo del nome. Questi metodi non vengono chiamati direttamente facendo inst.__metodo__, ma vengono in genere chiamati automaticamente in situazioni particolari

#Uno di questi metodi speciali è __init__, chiamato automaticamente ogni volta che un’istanza viene creata
#Quando un attributo di istanza viene dichiarato all’interno di un metodo (ad esempio l’__init__), 
#si usa self.attributo = valore, dato che il self si riferisce all’istanza:
# definiamo una classe che rappresenta un rettangolo generico
#
class Rectangle:
    def __init__(self, base, height):
        """Initialize the base and height attributes."""
        self.base = base
        self.height = height
    def calc_area(self):
        """Calculate and return the area of the rectangle."""
        return self.base * self.height
    def calc_perimeter(self):
        """Calculate and return the perimeter of a rectangle."""
        return (self.base + self.height) * 2

Rect=Rectangle(3,5)
print(Rect.base,Rect.height,Rect.calc_area(),Rect.calc_perimeter())


class Square(Rectangle):
    def __init__(self,base,height,name):
        """Initialize the base and height attributes."""
        super().__init__(base,height)
        if self.base==self.height:
            self.name = "quadrato"
        else:
            self.name="rettangolo"
    def print_attr(self):
        return (self.name,self.base,self.height)

        
sq=Square(3,3,'quadrato')
print(sq.base,sq.calc_area())
print('....................')
print(sq.print_attr())

"""from random import randrange
# creiamo una lista di 100 istanze di Rectangle con valori casuali
rects = [Rectangle(randrange(100), randrange(100)) for x in range(100)]
# iteriamo la lista di rettangoli e printiamo
# base, altezza, area, perimetro di ogni rettangolo
for rect in rects:
    print('Rect:', rect.base, rect.height)
    print('  Area:', rect.calc_area())
    print('  Perimeter:', rect.calc_perimeter())"""




