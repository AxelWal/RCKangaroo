package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"rckangaroo/fastbase"
	"encoding/hex"
)

func main() {
	// Parse command line arguments
	filename := flag.String("file", "", "Path to the FastBase file to load")
	prefix := flag.String("prefix", "", "Show records with this 3-byte prefix (format: 00f1f5)")
	flag.Parse()

	if *filename == "" {
		fmt.Println("Error: Please provide a FastBase file path using -file flag")
		flag.Usage()
		os.Exit(1)
	}

	// Ensure file exists
	if _, err := os.Stat(*filename); os.IsNotExist(err) {
		fmt.Printf("Error: File '%s' does not exist\n", *filename)
		os.Exit(1)
	}

	// Create new FastBase instance
	fb := fastbase.NewFastBase()

	// Load the file
	fmt.Printf("Loading FastBase file: %s\n", *filename)
	err := fb.LoadFromFile(*filename)
	if err != nil {
		fmt.Printf("Error loading FastBase file: %v\n", err)
		os.Exit(1)
	}

	// If prefix is specified, show only those records
	if *prefix != "" {
		if err := showRecordsByPrefix(fb, *prefix); err != nil {
			fmt.Printf("Error: %v\n", err)
			os.Exit(1)
		}
		return
	}

	// Otherwise show general statistics
	printStats(fb)
}

func printStats(fb *fastbase.FastBase) {
	fmt.Printf("\nFastBase Statistics for %s:\n", filepath.Base(flag.Arg(0)))
	fmt.Printf("----------------------------------------\n")

	// Calculate statistics
	totalLists := 256 * 256 * 256
	nonEmptyLists := 0
	totalRecords := 0
	maxListSize := uint16(0)
	var maxListPrefix [3]byte

	// Track kangaroo counts and their largest lists
	kangCounts := [3]int{0, 0, 0} // tame, wild1, wild2
	maxKangListSizes := [3]uint16{0, 0, 0}
	var maxKangListPrefixes [3][3]byte

	for i := 0; i < 256; i++ {
		for j := 0; j < 256; j++ {
			for k := 0; k < 256; k++ {
				list := fb.Lists[i][j][k]
				if list.Count > 0 {
					nonEmptyLists++
					totalRecords += int(list.Count)
					if list.Count > maxListSize {
						maxListSize = list.Count
						maxListPrefix = [3]byte{byte(i), byte(j), byte(k)}
					}

					// Count kangaroos by type in this list
					typeCountsInList := [3]uint16{0, 0, 0}
					for m := uint16(0); m < list.Count; m++ {
						ptr := list.Data[m]
						mem := fb.Pools[i].GetRecordPtr(ptr)
						kangType := mem[31]
						if kangType < 3 {
							typeCountsInList[kangType]++
							kangCounts[kangType]++
						}
					}

					// Update max lists for each type
					for t := 0; t < 3; t++ {
						if typeCountsInList[t] > maxKangListSizes[t] {
							maxKangListSizes[t] = typeCountsInList[t]
							maxKangListPrefixes[t] = [3]byte{byte(i), byte(j), byte(k)}
						}
					}
				}
			}
		}
	}

	// Print general statistics
	fmt.Printf("Total Lists:          %d\n", totalLists)
	fmt.Printf("Non-empty Lists:      %d (%.2f%%)\n", nonEmptyLists, float64(nonEmptyLists)*100/float64(totalLists))
	fmt.Printf("Total Records:        %d\n", totalRecords)
	fmt.Printf("Average Records/List: %.2f\n", float64(totalRecords)/float64(nonEmptyLists))
	fmt.Printf("Max List Size:        %d\n", maxListSize)
	fmt.Printf("Max List Prefix:      [%02x %02x %02x]\n", maxListPrefix[0], maxListPrefix[1], maxListPrefix[2])

	// Print kangaroo type statistics
	fmt.Printf("\nKangaroo Type Statistics:\n")
	fmt.Printf("----------------------------------------\n")
	kangTypes := []string{"Tame", "Wild1", "Wild2"}
	for t := 0; t < 3; t++ {
		fmt.Printf("%s Kangaroos:      %d\n", kangTypes[t], kangCounts[t])
		if kangCounts[t] > 0 {
			fmt.Printf("  Largest List:     %d points at [%02x %02x %02x]\n",
				maxKangListSizes[t],
				maxKangListPrefixes[t][0],
				maxKangListPrefixes[t][1],
				maxKangListPrefixes[t][2])
		}
	}

	// Print records in largest list
	fmt.Printf("\nRecords in largest list (Kangaroo Algorithm Points):\n")
	fmt.Printf("----------------------------------------\n")
	fmt.Printf("Format: Each 32-byte record contains:\n")
	fmt.Printf("- x[12]: x-coordinate on secp256k1 curve (compressed)\n")
	fmt.Printf("- d[19]: distance value in kangaroo algorithm\n")
	fmt.Printf("- type[1]: point type (0=tame, 1=wild1, 2=wild2)\n")
	fmt.Printf("\nKey Derivation:\n")
	fmt.Printf("1. For tame points (type=0):\n")
	fmt.Printf("   privKey = tame_distance - wild_distance + Int_HalfRange\n")
	fmt.Printf("2. For wild points (type=1,2):\n")
	fmt.Printf("   privKey = (tame_distance - wild_distance)/2 + Int_HalfRange\n")
	fmt.Printf("3. Verify solution:\n")
	fmt.Printf("   - P = G * privKey (where G is secp256k1 generator)\n")
	fmt.Printf("   - Check if P.x matches record's x-coordinate\n")
	fmt.Printf("----------------------------------------\n")

	// Print each record in the largest list
	list := fb.Lists[maxListPrefix[0]][maxListPrefix[1]][maxListPrefix[2]]
	for i := uint16(0); i < list.Count; i++ {
		ptr := list.Data[i]
		mem := fb.Pools[maxListPrefix[0]].GetRecordPtr(ptr)

		fmt.Printf("Record %d:\n", i+1)
		fmt.Printf("  x-coordinate: %02x%02x%02x%02x %02x%02x%02x%02x %02x%02x%02x%02x \n",
			mem[0], mem[1], mem[2], mem[3], mem[4], mem[5], mem[6], mem[7], mem[8], mem[9], mem[10], mem[11])
		fmt.Printf("  distance:     %02x%02x%02x%02x %02x%02x%02x%02x %02x%02x%02x%02x %02x%02x%02x%02x %02x%02x%02x\n",
			mem[12], mem[13], mem[14], mem[15], mem[16], mem[17], mem[18], mem[19], mem[20], mem[21], mem[22], mem[23],
			mem[24], mem[25], mem[26], mem[27], mem[28], mem[29], mem[30])
		fmt.Printf("  type:         %d (%s)\n", mem[31], getPointTypeName(mem[31]))
		fmt.Printf("----------------------------------------\n")
	}
}

func getPointTypeName(pointType byte) string {
	switch pointType {
	case 0:
		return "tame"
	case 1:
		return "wild1"
	case 2:
		return "wild2"
	default:
		return "unknown"
	}
}

func parsePrefix(prefix string) ([3]byte, error) {
	var result [3]byte
	
	// Remove any spaces and 0x prefix
	prefix = strings.ReplaceAll(prefix, " ", "")
	prefix = strings.TrimPrefix(prefix, "0x")
	
	if len(prefix) != 6 {
		return result, fmt.Errorf("prefix must be exactly 6 hex characters (3 bytes), got %d characters", len(prefix))
	}

	// Convert from hex
	decoded, err := hex.DecodeString(prefix)
	if err != nil {
		return result, fmt.Errorf("invalid hex string: %v", err)
	}

	copy(result[:], decoded)
	return result, nil
}

func showRecordsByPrefix(fb *fastbase.FastBase, prefixStr string) error {
	// Parse the prefix
	prefix, err := parsePrefix(prefixStr)
	if err != nil {
		return err
	}

	// Get the list for this prefix
	list := fb.Lists[prefix[0]][prefix[1]][prefix[2]]
	
	fmt.Printf("\nRecords with prefix [%02x %02x %02x]:\n", prefix[0], prefix[1], prefix[2])
	fmt.Printf("Total records: %d\n", list.Count)
	fmt.Printf("----------------------------------------\n")
	
	if list.Count == 0 {
		return nil
	}

	// Print format information
	fmt.Printf("Format: Each 32-byte record contains:\n")
	fmt.Printf("- x[12]: x-coordinate on secp256k1 curve (compressed)\n")
	fmt.Printf("- d[19]: distance value in kangaroo algorithm\n")
	fmt.Printf("- type[1]: point type (0=tame, 1=wild1, 2=wild2)\n")
	fmt.Printf("----------------------------------------\n")

	// Print each record
	for i := uint16(0); i < list.Count; i++ {
		ptr := list.Data[i]
		mem := fb.Pools[prefix[0]].GetRecordPtr(ptr)

		fmt.Printf("Record %d:\n", i+1)
		fmt.Printf("  x-coordinate: %02x%02x%02x%02x %02x%02x%02x%02x %02x%02x%02x%02x \n",
			mem[0], mem[1], mem[2], mem[3], mem[4], mem[5], mem[6], mem[7], mem[8], mem[9], mem[10], mem[11])
		fmt.Printf("  distance:     %02x%02x%02x%02x %02x%02x%02x%02x %02x%02x%02x%02x %02x%02x%02x%02x %02x%02x%02x\n",
			mem[12], mem[13], mem[14], mem[15], mem[16], mem[17], mem[18], mem[19], mem[20], mem[21], mem[22], mem[23],
			mem[24], mem[25], mem[26], mem[27], mem[28], mem[29], mem[30])
		fmt.Printf("  type:         %d (%s)\n", mem[31], getPointTypeName(mem[31]))
		fmt.Printf("----------------------------------------\n")
	}

	return nil
}
