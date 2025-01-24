// Package fastbase implements fast data storage and retrieval for the RCKangaroo algorithm
package fastbase

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
)

const (
	// MemPageSize represents the size of a memory page
	MemPageSize = 1 << 20 // 1MB

	// MaxPageCount is the maximum number of pages allowed in a memory pool
	MaxPageCount = 1 << 16 // 64K pages

	// DBRecordLength is the length of each data block record
	DBRecordLength = 32

	// DBMinGrowCount is the minimum growth count for list capacity
	DBMinGrowCount = 16

	// DBFindLength is the length used for data block comparison
	DBFindLength = 29

	// RecordsPerPage is the number of records that can fit in a memory page
	RecordsPerPage = MemPageSize / DBRecordLength
)

// MaxListSize is the maximum number of items allowed in a single list
const MaxListSize uint16 = 0xFFFF // Allow maximum uint16 value

// ListRecord represents a list of data block references
type ListRecord struct {
	Count    uint16   // Number of items in the list
	Capacity uint16   // Allocated capacity
	Data     []uint32 // References to data blocks
}

// MemPool manages memory allocation for data blocks
type MemPool struct {
	Pages [][]byte // Memory pages
	Ptr   uint32   // Current pointer position in the current page
}

// FastBase implements a fast storage and retrieval system using prefix-based indexing
type FastBase struct {
	Pools  [256]MemPool               // Memory pools for each first byte prefix
	Lists  [256][256][256]*ListRecord // 3-byte prefix based lookup table
	Header [256]byte                  // Header information
}

// NewFastBase creates a new FastBase instance
func NewFastBase() *FastBase {
	fb := &FastBase{}

	// Initialize all list records
	for i := 0; i < 256; i++ {
		for j := 0; j < 256; j++ {
			for k := 0; k < 256; k++ {
				fb.Lists[i][j][k] = &ListRecord{}
			}
		}
	}

	return fb
}

// Clear removes all data from the FastBase
func (fb *FastBase) Clear() {
	// Clear all memory pools
	for i := range fb.Pools {
		fb.Pools[i].Pages = fb.Pools[i].Pages[:0]
		fb.Pools[i].Ptr = 0
	}

	// Reset all lists
	for i := 0; i < 256; i++ {
		for j := 0; j < 256; j++ {
			for k := 0; k < 256; k++ {
				fb.Lists[i][j][k].Count = 0
				fb.Lists[i][j][k].Capacity = 0
				fb.Lists[i][j][k].Data = nil
			}
		}
	}
}

// AddDataBlock adds a new data block to the FastBase
func (fb *FastBase) AddDataBlock(data []byte, pos int) ([]byte, error) {
	if len(data) < 3 {
		return nil, errors.New("data block must be at least 3 bytes")
	}

	// Get the list for the 3-byte prefix
	list := fb.Lists[data[0]][data[1]][data[2]]

	// Allocate memory for the data block
	ptr, mem, err := fb.Pools[data[0]].allocRecord()
	if err != nil {
		return nil, err
	}

	// Copy data to the allocated memory
	copy(mem, data[3:])

	// Insert the pointer into the list at the specified position
	if pos == -1 {
		pos = int(list.Count)
	}

	// Ensure capacity
	if list.Count >= list.Capacity {
		grow := uint16(list.Count / 2)
		if grow < DBMinGrowCount {
			grow = DBMinGrowCount
		}
		newCap := list.Count + grow
		if newCap > 0xFFFF {
			return nil, errors.New("list capacity overflow")
		}

		newData := make([]uint32, newCap)
		copy(newData, list.Data)
		list.Data = newData
		list.Capacity = newCap
	}

	// Insert the pointer
	if pos < int(list.Count) {
		copy(list.Data[pos+1:], list.Data[pos:list.Count])
	}
	list.Data[pos] = ptr
	list.Count++

	return mem, nil
}

// FindDataBlock searches for a data block in the FastBase
func (fb *FastBase) FindDataBlock(data []byte) []byte {
	if len(data) < 3 {
		return nil
	}

	list := fb.Lists[data[0]][data[1]][data[2]]
	pos := fb.lowerBound(list, data[0], data[3:])

	if pos >= int(list.Count) {
		return nil
	}

	ptr := list.Data[pos]
	mem := fb.Pools[data[0]].GetRecordPtr(ptr)

	// Compare the data
	for i := 0; i < DBFindLength; i++ {
		if mem[i] != data[i+3] {
			return nil
		}
	}

	return mem
}

// SaveToFile saves the FastBase to a file
func (fb *FastBase) SaveToFile(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// Write header
	if _, err := file.Write(fb.Header[:]); err != nil {
		return err
	}

	// Write lists
	for i := 0; i < 256; i++ {
		for j := 0; j < 256; j++ {
			for k := 0; k < 256; k++ {
				list := fb.Lists[i][j][k]
				if err := binary.Write(file, binary.LittleEndian, list.Count); err != nil {
					return err
				}
				if list.Count > 0 {
					if err := binary.Write(file, binary.LittleEndian, list.Data[:list.Count]); err != nil {
						return err
					}
				}
			}
		}
	}

	return nil
}

// LoadFromFile loads the FastBase from a file
func (fb *FastBase) LoadFromFile(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	fb.Clear()

	// Read header
	if _, err := file.Read(fb.Header[:]); err != nil {
		return fmt.Errorf("error reading header: %v", err)
	}

	// Read lists
	for i := 0; i < 256; i++ {
		for j := 0; j < 256; j++ {
			for k := 0; k < 256; k++ {
				list := fb.Lists[i][j][k]
				var count uint16
				if err := binary.Read(file, binary.LittleEndian, &count); err != nil {
					if err == io.EOF {
						return fmt.Errorf("unexpected EOF at position [%d][%d][%d]", i, j, k)
					}
					return fmt.Errorf("error reading count at [%d][%d][%d]: %v", i, j, k, err)
				}

				list.Count = count
				if count > 0 {
					// Calculate capacity with growth factor
					grow := uint16(count / 2)
					if grow < DBMinGrowCount {
						grow = DBMinGrowCount
					}
					newCap := count + grow
					if newCap > MaxListSize {
						newCap = MaxListSize
					}

					// Allocate slice for data pointers
					list.Data = make([]uint32, count, newCap)
					list.Capacity = newCap

					// Read each data block
					for m := uint16(0); m < count; m++ {
						// Allocate memory for the data block
						ptr, mem, err := fb.Pools[i].allocRecord()
						if err != nil {
							return fmt.Errorf("error allocating memory at [%02x][%02x][%02x]: %v", i, j, k, err)
						}
						
						// Store the pointer
						list.Data[m] = ptr

						// Read the data block
						if _, err := io.ReadFull(file, mem); err != nil {
							return fmt.Errorf("error reading data block at [%02x][%02x][%02x]: %v", i, j, k, err)
						}
					}
				}
			}
		}
	}

	return nil
}

// allocRecord allocates a new record in the memory pool
func (mp *MemPool) allocRecord() (uint32, []byte, error) {
	if len(mp.Pages) == 0 || mp.Ptr+DBRecordLength > MemPageSize {
		if len(mp.Pages) >= MaxPageCount {
			return 0, nil, errors.New("memory pool overflow")
		}
		mp.Pages = append(mp.Pages, make([]byte, MemPageSize))
		mp.Ptr = 0
	}

	pageIndex := len(mp.Pages) - 1
	mem := mp.Pages[pageIndex][mp.Ptr : mp.Ptr+DBRecordLength]
	ptr := uint32(pageIndex*RecordsPerPage) | uint32(mp.Ptr/DBRecordLength)
	mp.Ptr += DBRecordLength

	return ptr, mem, nil
}

// GetRecordPtr returns a pointer to the record data for a given pointer value
func (mp *MemPool) GetRecordPtr(ptr uint32) []byte {
	pageIndex := ptr / uint32(RecordsPerPage)
	offset := (ptr % uint32(RecordsPerPage)) * DBRecordLength
	return mp.Pages[pageIndex][offset : offset+DBRecordLength]
}

// lowerBound performs a binary search to find the insertion point for a data block
func (fb *FastBase) lowerBound(list *ListRecord, poolIndex byte, data []byte) int {
	left, right := 0, int(list.Count)

	for left < right {
		mid := (left + right) / 2
		ptr := list.Data[mid]
		mem := fb.Pools[poolIndex].GetRecordPtr(ptr)

		// Compare data
		cmp := 0
		for i := 0; i < DBFindLength && cmp == 0; i++ {
			cmp = int(mem[i]) - int(data[i])
		}

		if cmp < 0 {
			left = mid + 1
		} else {
			right = mid
		}
	}

	return left
}
