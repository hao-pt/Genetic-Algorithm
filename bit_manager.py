import sys
from bitstring import BitArray, Bits

class Number:
    """Number class encode by binary bits"""
    def __init__(self, data=0, no_bits=6):
        self.data = BitArray(int=data, length=no_bits)
        self.no_bits = no_bits
    
    def to_bin(self) -> "bin_string":
        return self.data.bin
    
    def set_from_binary_string(self, bin_string: str):
        bin_string = bin_string.rjust(self.no_bits, "0")
        self.data = BitArray(bin=bin_string)

    def to_int(self) -> int:
        return self.data.int

    def __len__(self) -> int:
        """Return length of bits"""
        return len(self.data)

    def __getitem__(self, i: "int/slice") -> "1/0":
        return self.data[i]

    def __setitem__(self, i: "int/slice", value: "int/list"):
        self.data[i] = value

    def getsizeof(self) -> int:
        """Get size of data (Byte)"""
        return sys.getsizeof(self.data)

    def __str__(self):
        return str(self.data.int)

    def __delitem__(self, i: "int/slice"):
        del self.data[i]
        self.no_bits = self.__len__()


class NumberArray(Number):
    """NumberArray class construct array by Number class (each item encoded by bits)"""
    def __init__(self, no_bits_per_item=6, no_items=4):
        self.no_bits = no_bits_per_item * no_items # Compute total bits present array
        self.no_bits_per_item = no_bits_per_item
        self.no_items = no_items
        super().__init__(no_bits=self.no_bits)
    
    def __getitem__(self, i: int):
        if isinstance(i, slice):
            start = 0 if i.start == None else i.start
            stop = self.no_items if i.stop == None else i.stop
            step = 1 if i.step == None else i.step

            if stop < 0:
                stop = self.no_items + stop
            if start < 0:
                start = self.no_items + start
                if start > stop and step > 0:
                    step = -1

            return [self.data[j*self.no_bits_per_item:(j+1)*self.no_bits_per_item].int for j in range(start, stop, step)]
        if i < 0:
            i = i + self.no_items
        return self.data[i*self.no_bits_per_item:(i+1)*self.no_bits_per_item].int

    def __setitem__(self, i: 'int/slice', value: 'int/list]'):
        if isinstance(i, slice):
            start = 0 if i.start == None else i.start
            stop = self.no_items if i.stop == None else i.stop
            step = 1 if i.step == None else i.step

            if stop < 0:
                stop = self.no_items + stop
            if start < 0:
                start = self.no_items + start
                if start > stop and step > 0:
                    step = -1
            
            if isinstance(value, int):
                raise ValueError("can only assign an iterable")
            else:
                index = 0
                for j in range(start, stop, step):
                    if index < len(value):
                        num = value[index]
                        if isinstance(value[index], Number):
                            num = value[index].to_int()
                        self.data[j*self.no_bits_per_item:(j+1)*self.no_bits_per_item] = BitArray(int=num, length=self.no_bits_per_item)
                        index += 1
                    else:
                        del self.data[j*self.no_bits_per_item:(j+1)*self.no_bits_per_item]
                # Update
                self.no_items = self.__len__()
                self.no_bits = self.no_items * self.no_bits_per_item
                
        else:
            if i < 0:
                i = i + self.no_items   
            if isinstance(value, Number):
                value = value.to_int()
            self.data[i*self.no_bits_per_item:(i+1)*self.no_bits_per_item] = BitArray(int=value, length=self.no_bits_per_item)

    def __eq__(self, other):
        if isinstance(other, list):
            for i in range(len(list)):
                self.data[i*self.no_bits_per_item:(i+1)*self.no_bits_per_item] = BitArray(int=other[i], length=self.no_bits_per_item)

    def __len__(self):
        return super().__len__() // self.no_bits_per_item

    def __delitem__(self, i: 'int/slice'):
        if isinstance(i, slice):
            start = 0 if i.start == None else i.start
            stop = self.no_items if i.stop == None else i.stop
            step = 1 if i.step == None else i.step

            if stop < 0:
                stop = self.no_items + stop
            if start < 0:
                start = self.no_items + start
                if start > stop and step > 0:
                    step = -1
            
            j = start # Loop condition
            index = start # index of array

            while j < stop:
                del self.data[index*self.no_bits_per_item:(index+1)*self.no_bits_per_item]
                j += step
                index += step - 1
                
        else:
            del self.data[i*self.no_bits_per_item:(i+1)*self.no_bits_per_item]

        # Update
        self.no_items = self.__len__()
        self.no_bits = self.no_items * self.no_bits_per_item

    def __str__(self):
        list_num = []
        for i in range(self.no_items):
            item = self.data[i*self.no_bits_per_item:(i+1)*self.no_bits_per_item].int
            list_num.append(item)
        return str(list_num)

    def __iter__(self):
        num_list = self.__getitem__(slice(0, self.no_items))
        return iter(num_list)
                
            




# a = BitArray(bin="001000110100")
# print(a.bin)
# print(sys.getsizeof(a))

# b = Number(3)
# print(b.to_bin())
# b[0:2] = [1, 0]
# print(b.to_bin())
# print(type(b[3]))
# b[0] = False
# print(b.to_bin())
# print(len(b))
# a = NumberArray()
# a[1:4] = [b, Number(5)]
# a[0] = 4




# c = Number(29424828428424, no_bits=100)
# print(c.to_bin())
# print(c.length())
# print(c.getsizeof())

# a = NumberArray(no_items=5)
# a = [5, 6, 7, 8, 10, 12]
# print(len(a))
# print(a)
# del a[0:4:3]
# print(a)

# a = NumberArray()
# a = [31, 7, 38, 1]
# print(a)
# a = sorted(a, key=lambda x: x**2 - 10*x + 5, reverse=True)
# print(a)

# for num in a:
#     print(num)

# b = [1, 2, 3, 4]
# b[1:3] = [3]

# print(b)

# c = Number(64, no_bits=7)
# print(c.to_bin())
# print(c.to_int())
# del c[3:5]
# print(c.to_bin())
# print(c.to_int())

# d = NumberArray()
# d[:] = [3, 9, 7, 24]
# print(type(d))
# for num in d:
#     print(num)

# e = [num for num in d]
# print(e)

# a = NumberArray()
# a[:] = [1, 2, 3, 4]
# print(type(a))
# print(a[-1])

# b = [num for num in a]
# print(b)
# print(type(b))

# a.append("0b0111")
# for i in a:
#     print(i)

# a = NumberArray()
# a[1] = 63
# print(a)

# a = NumberArray()
# a[0] = -3
# print(bin(a[0]))

# equation = lambda x: 2**x + 3
# a = [1, 5, -3, -10, 20]
# b = sorted(a, key=equation+3)
# print(b)