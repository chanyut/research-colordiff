/*
reference CIEDE2000 from https://github.com/mattn/go-ciede2000
algorithm and equation reference http://www2.ece.rochester.edu/~gsharma/ciede2000/ciede2000noteCRNA.pdf
*/

package main

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/pkg/errors"
	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"

	"github.com/lucasb-eyer/go-colorful"
)

func main() {
	rand.Seed(time.Now().Unix())

	fandeckColors, err := loadCSV()
	if err != nil {
		panic(err)
	}

	// just for test...
	// generateHSVSpectrumImage(400, 200, "./colorspace.png")
	// generateHSVSpectrumImageWithFandeck(400, 200, fandeckColors, "./colorspace-fandeck.png")
	// return

	startedOn := time.Now()

	samp_R := randomUniqueUint8(10)
	samp_G := randomUniqueUint8(10)
	samp_B := randomUniqueUint8(100)

	const numberOfWorkers = 8
	numberOfJobs := len(samp_R) * len(samp_G)
	resultChan := make(chan []BenchmarkResult)
	jobsChan := make(chan []color.Color, numberOfJobs)
	doneChan := make(chan bool)

	// spin up workers...
	for w := 1; w <= numberOfWorkers; w++ {
		go benchmarkWorker(w, fandeckColors, jobsChan, resultChan)
	}

	// push benchmarking jobs...
	totalJobsIssued := 0
	for _, r := range samp_R {
		for _, g := range samp_G {
			samplings := []color.Color{}
			for _, b := range samp_B {
				samplings = append(samplings, color.RGBA{r, g, b, 255})
			}
			jobsChan <- samplings
			totalJobsIssued++
		}
	}
	close(jobsChan)
	log.Println("total jobs:", totalJobsIssued)

	// go routine for benchmarking result
	allResult := []BenchmarkResult{}
	go func() {
		totalDiff := 0
		for i := 1; i <= numberOfJobs; i++ {
			result := <-resultChan
			totalDiff += len(result)
			log.Println("job[", i, "/", numberOfJobs, "]... done")
			allResult = append(allResult, result...)
		}
		doneChan <- true
	}()

	<-doneChan
	log.Println("benchmarking done in", time.Since(startedOn))
	log.Println("started rendering result...")

	if err := makeBenchmarkResultCSVFile(allResult, "./benchmark.csv"); err != nil {
		panic(err)
	}

	if err := makeBenchmarkResultCSVFile(allResult, "benchmark.png"); err != nil {
		panic(err)
	}
}

func benchmarkWorker(id int, fandeckColors []FandeckColor, samplingChan <-chan []color.Color, resultChan chan<- []BenchmarkResult) {
	for samplings := range samplingChan {
		resultBuffer := []BenchmarkResult{}
		for _, samplingRGB := range samplings {
			result := benchmarkEach(samplingRGB, fandeckColors, false)
			if result.CompareDiffInRGBDistance > 0 {
				resultBuffer = append(resultBuffer, result)
			}
		}
		resultChan <- resultBuffer
	}
}

func benchmarkEach(samplingRGB color.Color, fandeckColors []FandeckColor, verbose bool) BenchmarkResult {
	diffRGB := math.MaxFloat64
	matchFDColorWithRGBDiff := fandeckColors[0]
	diffDeltaE := math.MaxFloat64
	matchFDColorWithDeltaEDiff := fandeckColors[0]

	for _, fdColor := range fandeckColors {
		dRGB := fdColor.DiffWithRGB(samplingRGB)
		if dRGB < diffRGB {
			diffRGB = dRGB
			matchFDColorWithRGBDiff = fdColor
		}

		dDeltaE := fdColor.DiffWithCIEDeltaE2000(samplingRGB)
		if dDeltaE < diffDeltaE {
			diffDeltaE = dDeltaE
			matchFDColorWithDeltaEDiff = fdColor
		}
	}

	if verbose {
		r_, g_, b_, _ := samplingRGB.RGBA()
		r := int(float64(r_) / math.MaxUint16 * 255)
		g := int(float64(g_) / math.MaxUint16 * 255)
		b := int(float64(b_) / math.MaxUint16 * 255)
		log.Printf("sampling RGB (%v, %v, %v)", r, g, b)
		log.Println("match with RGB Euclidean =>", matchFDColorWithRGBDiff.Debug())
		log.Println("match with DeltaE2000 =>", matchFDColorWithDeltaEDiff.Debug())
	}

	distance := matchFDColorWithRGBDiff.DiffWithRGB(matchFDColorWithDeltaEDiff.Color())

	result := BenchmarkResult{
		SamplingRGB:              samplingRGB,
		RGBEuclideanResult:       matchFDColorWithRGBDiff,
		DeltaEResult:             matchFDColorWithDeltaEDiff,
		CompareDiffInRGBDistance: distance,
	}

	if verbose {
		log.Println("compate distance:", result.CompareDiffInRGBDistance)
	}

	return result
}

type BenchmarkResult struct {
	SamplingRGB              color.Color
	RGBEuclideanResult       FandeckColor
	DeltaEResult             FandeckColor
	CompareDiffInRGBDistance float64
}

func (bench BenchmarkResult) ToCSV() string {
	r_, g_, b_, _ := bench.SamplingRGB.RGBA()
	r := int(float64(r_) / math.MaxUint16 * 255)
	g := int(float64(g_) / math.MaxUint16 * 255)
	b := int(float64(b_) / math.MaxUint16 * 255)

	eu := bench.RGBEuclideanResult
	de := bench.DeltaEResult

	hex2 := fmt.Sprintf("#%02x%02x%02x", r, g, b)
	hex1 := fmt.Sprintf("#%02x%02x%02x", eu.R, eu.G, eu.B)
	hex3 := fmt.Sprintf("#%02x%02x%02x", de.R, de.G, de.B)

	return fmt.Sprintf("%s;%v;%v;%v;%s;%s;%v;%v;%v;%s;%s;%v;%v;%v", hex1, r, g, b,
		eu.Code, hex2, eu.R, eu.G, eu.B,
		de.Code, hex3, de.R, de.G, de.B,
	)
}

func (bench BenchmarkResult) Debug() {
	r_, g_, b_, _ := bench.SamplingRGB.RGBA()
	r := int(float64(r_) / math.MaxUint16 * 255)
	g := int(float64(g_) / math.MaxUint16 * 255)
	b := int(float64(b_) / math.MaxUint16 * 255)
	log.Printf("sampling RGB (%v, %v, %v)", r, g, b)
	log.Println("match with RGB Euclidean =>", bench.RGBEuclideanResult.Debug())
	log.Println("match with DeltaE2000 =>", bench.DeltaEResult.Debug())
}

type FandeckColor struct {
	Code string
	R    uint8
	G    uint8
	B    uint8
}

func (c FandeckColor) Color() color.Color {
	return color.RGBA{c.R, c.G, c.B, 255}
}

func (c FandeckColor) Debug() string {
	return fmt.Sprintf("%s (%v, %v, %v)", c.Code, c.R, c.G, c.B)
}

// DiffWithRGB returns Euclidean distance implementation of RGB color
func (c FandeckColor) DiffWithRGB(other color.Color) float64 {
	r_, g_, b_, _ := other.RGBA()
	r := int(float64(r_) / math.MaxUint16 * 255)
	g := int(float64(g_) / math.MaxUint16 * 255)
	b := int(float64(b_) / math.MaxUint16 * 255)
	dr := float64(int(c.R) - r)
	dg := float64(int(c.G) - g)
	db := float64(int(c.B) - b)
	return math.Sqrt(dr*dr + dg*dg + db*db)
}

// DiffWithRGB returns Euclidean distance implementation of RGB color
func (c FandeckColor) DiffWithCIEDeltaE2000(other color.Color) float64 {
	return CIEDE2000(ToLAB(c.Color()), ToLAB(other))
}

type LAB struct {
	L float64
	A float64
	B float64
}

func ToXYZ(c color.Color) (float64, float64, float64) {
	ta, tg, tb, _ := c.RGBA()
	r := float64(ta) / 65535.0
	g := float64(tg) / 65535.0
	b := float64(tb) / 65535.0

	if r > 0.04045 {
		r = math.Pow(((r + 0.055) / 1.055), 2.4)
	} else {
		r = r / 12.92
	}

	if g > 0.04045 {
		g = math.Pow(((g + 0.055) / 1.055), 2.4)
	} else {
		g = g / 12.92
	}

	if b > 0.04045 {
		b = math.Pow(((b + 0.055) / 1.055), 2.4)
	} else {
		b = b / 12.92
	}

	r *= 100
	g *= 100
	b *= 100
	return r*0.4124 + g*0.3576 + b*0.1805, r*0.2126 + g*0.7152 + b*0.0722, r*0.0193 + g*0.1192 + b*0.9505
}

func ToLAB(c color.Color) *LAB {
	x, y, z := ToXYZ(c)
	x /= 95.047
	y /= 100.000
	z /= 108.883

	if x > 0.008856 {
		x = math.Pow(x, (1.0 / 3.0))
	} else {
		x = (7.787 * x) + (16 / 116)
	}

	if y > 0.008856 {
		y = math.Pow(y, (1.0 / 3.0))
	} else {
		y = (7.787 * y) + (16 / 116)
	}

	if z > 0.008856 {
		z = math.Pow(z, (1.0 / 3.0))
	} else {
		z = (7.787 * z) + (16 / 116)
	}

	l := (116 * y) - 16
	a := 500 * (x - y)
	b := 200 * (y - z)

	if l < 0.0 {
		l = 0.0
	}

	return &LAB{l, a, b}
}

func deg2Rad(deg float64) float64 {
	return deg * (math.Pi / 180.0)
}

func rad2Deg(rad float64) float64 {
	return (180.0 / math.Pi) * rad
}

func CIEDE2000(lab1, lab2 *LAB) float64 {
	/*
	 * "For these and all other numerical/graphical 􏰀delta E00 values
	 * reported in this article, we set the parametric weighting factors
	 * to unity(i.e., k_L = k_C = k_H = 1.0)." (Page 27).
	 */
	k_L, k_C, k_H := 1.0, 1.0, 1.0
	deg360InRad := deg2Rad(360.0)
	deg180InRad := deg2Rad(180.0)
	pow25To7 := 6103515625.0 /* pow(25, 7) */

	/*
	 * Step 1
	 */
	/* Equation 2 */
	C1 := math.Sqrt((lab1.A * lab1.A) + (lab1.B * lab1.B))
	C2 := math.Sqrt((lab2.A * lab2.A) + (lab2.B * lab2.B))
	/* Equation 3 */
	barC := (C1 + C2) / 2.0
	/* Equation 4 */
	G := 0.5 * (1 - math.Sqrt(math.Pow(barC, 7)/(math.Pow(barC, 7)+pow25To7)))
	/* Equation 5 */
	a1Prime := (1.0 + G) * lab1.A
	a2Prime := (1.0 + G) * lab2.A
	/* Equation 6 */
	CPrime1 := math.Sqrt((a1Prime * a1Prime) + (lab1.B * lab1.B))
	CPrime2 := math.Sqrt((a2Prime * a2Prime) + (lab2.B * lab2.B))
	/* Equation 7 */
	var hPrime1 float64
	if lab1.B == 0 && a1Prime == 0 {
		hPrime1 = 0.0
	} else {
		hPrime1 = math.Atan2(lab1.B, a1Prime)
		/*
		 * This must be converted to a hue angle in degrees between 0
		 * and 360 by addition of 2pi to negative hue angles.
		 */
		if hPrime1 < 0 {
			hPrime1 += deg360InRad
		}
	}
	var hPrime2 float64
	if lab2.B == 0 && a2Prime == 0 {
		hPrime2 = 0.0
	} else {
		hPrime2 = math.Atan2(lab2.B, a2Prime)
		/*
		 * This must be converted to a hue angle in degrees between 0
		 * and 360 by addition of 2􏰏 to negative hue angles.
		 */
		if hPrime2 < 0 {
			hPrime2 += deg360InRad
		}
	}

	/*
	 * Step 2
	 */
	/* Equation 8 */
	deltaLPrime := lab2.L - lab1.L
	/* Equation 9 */
	deltaCPrime := CPrime2 - CPrime1
	/* Equation 10 */
	var deltahPrime float64
	CPrimeProduct := CPrime1 * CPrime2
	if CPrimeProduct == 0 {
		deltahPrime = 0
	} else {
		/* Avoid the fabs() call */
		deltahPrime = hPrime2 - hPrime1
		if deltahPrime < -deg180InRad {
			deltahPrime += deg360InRad
		} else if deltahPrime > deg180InRad {
			deltahPrime -= deg360InRad
		}
	}
	/* Equation 11 */
	deltaHPrime := 2.0 * math.Sqrt(CPrimeProduct) * math.Sin(deltahPrime/2.0)

	/*
	 * Step 3
	 */
	/* Equation 12 */
	barLPrime := (lab1.L + lab2.L) / 2.0
	/* Equation 13 */
	barCPrime := (CPrime1 + CPrime2) / 2.0
	/* Equation 14 */
	var barhPrime float64
	hPrimeSum := hPrime1 + hPrime2
	if CPrime1*CPrime2 == 0 {
		barhPrime = hPrimeSum
	} else {
		if math.Abs(hPrime1-hPrime2) <= deg180InRad {
			barhPrime = hPrimeSum / 2.0
		} else {
			if hPrimeSum < deg360InRad {
				barhPrime = (hPrimeSum + deg360InRad) / 2.0
			} else {
				barhPrime = (hPrimeSum - deg360InRad) / 2.0
			}
		}
	}
	/* Equation 15 */
	T := 1.0 - (0.17 * math.Cos(barhPrime-deg2Rad(30.0))) +
		(0.24 * math.Cos(2.0*barhPrime)) +
		(0.32 * math.Cos((3.0*barhPrime)+deg2Rad(6.0))) -
		(0.20 * math.Cos((4.0*barhPrime)-deg2Rad(63.0)))
	/* Equation 16 */
	deltaTheta := deg2Rad(30.0) * math.Exp(-math.Pow((barhPrime-deg2Rad(275.0))/deg2Rad(25.0), 2.0))
	/* Equation 17 */
	R_C := 2.0 * math.Sqrt(math.Pow(barCPrime, 7.0)/(math.Pow(barCPrime, 7.0)+pow25To7))
	/* Equation 18 */
	S_L := 1 + ((0.015 * math.Pow(barLPrime-50.0, 2.0)) /
		math.Sqrt(20+math.Pow(barLPrime-50.0, 2.0)))
	/* Equation 19 */
	S_C := 1 + (0.045 * barCPrime)
	/* Equation 20 */
	S_H := 1 + (0.015 * barCPrime * T)
	/* Equation 21 */
	R_T := (-math.Sin(2.0 * deltaTheta)) * R_C

	/* Equation 22 */
	return math.Sqrt(
		math.Pow(deltaLPrime/(k_L*S_L), 2.0) +
			math.Pow(deltaCPrime/(k_C*S_C), 2.0) +
			math.Pow(deltaHPrime/(k_H*S_H), 2.0) +
			(R_T * (deltaCPrime / (k_C * S_C)) * (deltaHPrime / (k_H * S_H))))
}

func Diff(c1, c2 color.Color) float64 {
	return CIEDE2000(ToLAB(c1), ToLAB(c2))
}

func loadCSV() ([]FandeckColor, error) {
	csv, err := ioutil.ReadFile("./fandeck-colors.csv")
	if err != nil {
		return nil, errors.Wrap(err, "failed to read csv file")
	}

	fdColors := []FandeckColor{}
	lines := strings.Split(string(csv), "\n")
	for _, line := range lines {
		cols := strings.Split(line, ",")
		r, _ := strconv.Atoi(cols[2])
		g, _ := strconv.Atoi(cols[3])
		b, _ := strconv.Atoi(cols[4])

		fdColor := FandeckColor{
			Code: cols[0],
			R:    uint8(r),
			G:    uint8(g),
			B:    uint8(b),
		}
		fdColors = append(fdColors, fdColor)
	}

	return fdColors, nil
}

func randomUniqueUint8(n int) []uint8 {
	if n > 255 {
		panic("n is out of range (max: 255)")
	}
	r := []uint8{}
	for i := 0; i <= math.MaxUint8; i++ {
		r = append(r, uint8(i))
	}

	for i := 0; i < len(r); i++ {
		idx1 := rand.Intn(len(r))
		idx2 := rand.Intn(len(r))
		r[idx1], r[idx2] = r[idx2], r[idx1]
	}

	return r[:n]
}

func makeBenchmarkResultCSVFile(results []BenchmarkResult, outFilePath string) error {
	csv := strings.Builder{}
	csv.WriteString("r;g;b;euclidean_code;hex;r;g;b;deltae_code;hex;r;g;b\n")
	for _, res := range results {
		csv.WriteString(res.ToCSV())
		csv.WriteString("\n")
	}

	if _, err := os.Stat(outFilePath); os.IsExist(err) {
		os.Remove(outFilePath)
	}

	if err := ioutil.WriteFile(outFilePath, []byte(csv.String()), 0666); err != nil {
		return errors.Wrap(err, "failed to write csv file")
	}

	return nil
}

func makeBenchmarkResultPNGFile(results []BenchmarkResult, outFilePath string) error {
	img := image.NewRGBA(image.Rect(0, 0, 300, 44*len(results)))
	for idx, result := range results {
		x := 0
		y := idx * 44

		// draw sampling color
		r_, g_, b_, _ := result.SamplingRGB.RGBA()
		r := int(float64(r_) / math.MaxUint16 * 255)
		g := int(float64(g_) / math.MaxUint16 * 255)
		b := int(float64(b_) / math.MaxUint16 * 255)

		draw.Draw(img, image.Rect(x, y, x+100, y+44),
			&image.Uniform{result.SamplingRGB},
			image.Point{}, draw.Src)
		addLabel(img, x+4, y+24, fmt.Sprintf("#%02x%02x%02x", r, g, b))

		// draw euclidean match result
		x += 100
		draw.Draw(img, image.Rect(x, y, x+100, y+44),
			&image.Uniform{result.RGBEuclideanResult.Color()},
			image.Point{}, draw.Src)
		addLabel(img, x+4, y+24, result.RGBEuclideanResult.Code)

		// draw euclidean match result
		x += 100
		draw.Draw(img, image.Rect(x, y, x+100, y+44),
			&image.Uniform{result.DeltaEResult.Color()},
			image.Point{}, draw.Src)
		addLabel(img, x+4, y+24, result.DeltaEResult.Code)
	}
	f, err := os.Create(outFilePath)
	if err != nil {
		return errors.Wrap(err, "failed to create file")
	}
	defer f.Close()
	if err := png.Encode(f, img); err != nil {
		return errors.Wrap(err, "failed to encode PNG")
	}

	return nil
}

func addLabel(img *image.RGBA, x, y int, label string) {
	col := color.RGBA{0, 0, 0, 255}
	point := fixed.Point26_6{fixed.Int26_6(x * 64), fixed.Int26_6(y * 64)}
	d := &font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(col),
		Face: basicfont.Face7x13,
		Dot:  point,
	}
	d.DrawString(label)
}

func generateHSVSpectrumImage(width, height int, outFilePath string) error {
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for x := 0; x < width; x++ {
		ux := float64(x) / float64(width)
		h := 360.0 * ux
		for y := 0; y < height; y++ {
			uy := float64(y) / float64(height)
			s := 1.0
			v := 1.0
			if uy < 0.5 { // upper half
				s = uy * 2.0 // make sat from 0 - 1 along with increasing y
				v = 1.0      // keep value to 1
			} else {
				s = 1.0                      // keep sat to 1
				v = 1.0 - ((uy - 0.5) * 2.0) // make value from 1 - 0 along with increasing y
			}
			c := colorful.Hsv(h, s, v)
			r, g, b := c.RGB255()
			img.SetRGBA(x, y, color.RGBA{r, g, b, 255})
		}
	}

	f, err := os.Create(outFilePath)
	if err != nil {
		return errors.Wrap(err, "failed to create file")
	}
	defer f.Close()
	if err := png.Encode(f, img); err != nil {
		return errors.Wrap(err, "failed to encode PNG")
	}

	return nil
}

func generateHSVSpectrumImageWithFandeck(width, height int, fandeckColors []FandeckColor, outFilePath string) error {
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for x := 0; x < width; x++ {
		ux := float64(x) / float64(width)
		h := 360.0 * ux
		log.Println("rasterize...", ux*100, "%")
		for y := 0; y < height; y++ {
			uy := float64(y) / float64(height)
			s := 1.0
			v := 1.0
			if uy < 0.5 { // upper half
				s = uy * 2.0 // make sat from 0 - 1 along with increasing y
				v = 1.0      // keep value to 1
			} else {
				s = 1.0                      // keep sat to 1
				v = 1.0 - ((uy - 0.5) * 2.0) // make value from 1 - 0 along with increasing y
			}
			c := colorful.Hsv(h, s, v)
			r, g, b := c.RGB255()
			fdColor := findFandeckColor(color.RGBA{r, g, b, 255}, fandeckColors)
			img.Set(x, y, fdColor.Color())
		}
	}

	f, err := os.Create(outFilePath)
	if err != nil {
		return errors.Wrap(err, "failed to create file")
	}
	defer f.Close()
	if err := png.Encode(f, img); err != nil {
		return errors.Wrap(err, "failed to encode PNG")
	}

	return nil
}

func findFandeckColor(rgb color.Color, fandeckColors []FandeckColor) FandeckColor {
	diffDeltaE := math.MaxFloat64
	matchFDColorWithDeltaEDiff := fandeckColors[0]

	for _, fdColor := range fandeckColors {

		dDeltaE := fdColor.DiffWithCIEDeltaE2000(rgb)
		if dDeltaE < diffDeltaE {
			diffDeltaE = dDeltaE
			matchFDColorWithDeltaEDiff = fdColor
		}
	}

	return matchFDColorWithDeltaEDiff
}
